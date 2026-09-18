# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import base64
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.ai.ml.entities import (
    BatchDeployment,
    BuildContext,
    CodeConfiguration,
    Environment,
)
from azure.core.exceptions import ResourceNotFoundError

try:
    from azure.ai.ml._utils._registry_utils import (
        get_storage_details_for_registry_assets,
    )
except ImportError:
    get_storage_details_for_registry_assets = None

from aml_client import (
    add_workspace_arguments,
    create_ml_client,
    create_registry_ml_client,
    get_registered_model,
    wait_for_resource_create_or_update,
)

DEFAULT_BATCH_ENVIRONMENT = (
    "azureml://registries/azureml/environments/sklearn-1.5/versions/53"
)
IMMUTABLE_ENVIRONMENT_PATTERNS = (
    re.compile(r"azureml://registries/[^/\s]+/environments/[^/\s]+/versions/\d+"),
    re.compile(r"azureml:[^:\s]+:\d+"),
    re.compile(
        r"/subscriptions/[^/\s]+/resourceGroups/[^/\s]+/providers/"
        r"Microsoft\.MachineLearningServices/(?:workspaces|registries)/[^/\s]+/"
        r"environments/[^/\s]+/versions/\d+",
        re.IGNORECASE,
    ),
)
REGISTRY_ENVIRONMENT_PATTERN = re.compile(
    r"azureml://registries/([^/\s]+)/environments/([^/\s]+)/versions/(\d+)",
    re.IGNORECASE,
)
FULL_REGISTRY_ENVIRONMENT_PATTERN = re.compile(
    r"/subscriptions/([^/\s]+)/resourceGroups/([^/\s]+)/providers/"
    r"Microsoft\.MachineLearningServices/registries/([^/\s]+)/environments/"
    r"([^/\s]+)/versions/(\d+)",
    re.IGNORECASE,
)
FULL_ENVIRONMENT_ID_PATTERN = re.compile(
    r"/subscriptions/[^/\s]+/resourceGroups/[^/\s]+/providers/"
    r"Microsoft\.MachineLearningServices/(?:workspaces|registries)/[^/\s]+/"
    r"environments/[^/\s]+/versions/\d+",
    re.IGNORECASE,
)
SUBSCRIPTION_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
ARM_RESOURCE_NAME_PATTERN = re.compile(r"[A-Za-z0-9._()-]+")
STORAGE_CONTAINER_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])?")
IMMUTABLE_IMAGE_PATTERN = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}", re.IGNORECASE)
SUPPORTED_AZURE_AI_ML_VERSION = "1.30.0"
REGISTRY_ASSET_TYPE = "environments"
MAX_REGISTRY_BUILD_FILES = 100
MAX_REGISTRY_BUILD_FILE_BYTES = 32 * 1024 * 1024
MAX_REGISTRY_BUILD_TOTAL_BYTES = 128 * 1024 * 1024
MAX_REGISTRY_LISTING_BYTES = 2 * 1024 * 1024
SOURCE_REGISTRY_PROPERTY = "source_registry"
SOURCE_ENVIRONMENT_PROPERTY = "source_environment"
SOURCE_VERSION_PROPERTY = "source_version"
SOURCE_REFERENCE_PROPERTY = "source_reference"
SOURCE_MANIFEST_PROPERTY = "source_manifest_sha256"
SOURCE_PROVENANCE_PROPERTIES = (
    SOURCE_REGISTRY_PROPERTY,
    SOURCE_ENVIRONMENT_PROPERTY,
    SOURCE_VERSION_PROPERTY,
    SOURCE_REFERENCE_PROPERTY,
    SOURCE_MANIFEST_PROPERTY,
)


def validate_immutable_environment_reference(reference: str) -> str:
    normalized_reference = reference.strip()
    if not normalized_reference:
        raise argparse.ArgumentTypeError(
            "An immutable Azure ML environment reference is required."
        )
    if not any(
        pattern.fullmatch(normalized_reference)
        for pattern in IMMUTABLE_ENVIRONMENT_PATTERNS
    ):
        raise argparse.ArgumentTypeError(
            "Azure ML environment must be an immutable numeric version reference: "
            "'azureml://registries/<registry>/environments/<name>/versions/<version>', "
            "'azureml:<name>:<version>', or a full Azure resource ID ending in "
            "'/environments/<name>/versions/<version>'. Mutable labels, 'latest', "
            "unversioned references, images, and inline Conda environments are not "
            "allowed."
        )
    return normalized_reference


def _parse_registry_environment_reference(
    reference: str,
) -> tuple[str | None, str | None, str, str, str] | None:
    registry_uri_match = REGISTRY_ENVIRONMENT_PATTERN.fullmatch(reference)
    if registry_uri_match:
        registry_name, environment_name, version = registry_uri_match.groups()
        return None, None, registry_name, environment_name, version
    full_id_match = FULL_REGISTRY_ENVIRONMENT_PATTERN.fullmatch(reference)
    if full_id_match:
        return full_id_match.groups()
    return None


def _get_registry_environment(
    ml_client: object,
    reference: str,
) -> tuple[object, object, str, str, str, str]:
    parsed_reference = _parse_registry_environment_reference(reference)
    if not parsed_reference:
        raise RuntimeError("Expected an immutable registry environment reference.")

    (
        requested_subscription_id,
        requested_resource_group,
        registry_name,
        environment_name,
        version,
    ) = parsed_reference
    for field_name, value in (
        ("registry", registry_name),
        ("environment", environment_name),
    ):
        if not ARM_RESOURCE_NAME_PATTERN.fullmatch(value):
            raise RuntimeError(
                f"Immutable registry environment reference contains an unsafe "
                f"{field_name} name."
            )
    registry_client = create_registry_ml_client(ml_client, registry_name)
    try:
        environment = registry_client.environments.get(
            name=environment_name,
            version=version,
        )
    except Exception as exc:
        raise RuntimeError(
            "Immutable registry environment "
            f"'{registry_name}/{environment_name}:{version}' was not found."
        ) from exc

    operation_scope = getattr(registry_client.environments, "_operation_scope", None)
    subscription_id = str(getattr(operation_scope, "subscription_id", "") or "")
    resource_group_name = str(getattr(operation_scope, "resource_group_name", "") or "")
    scoped_registry_name = str(getattr(operation_scope, "registry_name", "") or "")
    if not SUBSCRIPTION_ID_PATTERN.fullmatch(subscription_id):
        raise RuntimeError(
            "Azure ML registry operation scope returned a missing or invalid "
            "subscription ID."
        )
    for field_name, value in (
        ("resource group", resource_group_name),
        ("registry", scoped_registry_name),
        ("environment", environment_name),
    ):
        if not ARM_RESOURCE_NAME_PATTERN.fullmatch(value):
            raise RuntimeError(
                "Azure ML registry operation scope returned a missing or unsafe "
                f"{field_name} name."
            )
    if scoped_registry_name.lower() != registry_name.lower():
        raise RuntimeError(
            "Azure ML registry operation scope did not match the requested "
            f"registry. requested={registry_name!r}, "
            f"resolved={scoped_registry_name!r}"
        )
    if requested_subscription_id and (
        subscription_id.lower() != requested_subscription_id.lower()
        or resource_group_name.lower() != str(requested_resource_group).lower()
    ):
        raise RuntimeError(
            "Azure ML registry operation scope did not match the requested full "
            "registry resource ID."
        )

    resolved_id = (
        f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/"
        "providers/Microsoft.MachineLearningServices/registries/"
        f"{scoped_registry_name}/environments/{environment_name}/versions/{version}"
    )
    if not FULL_ENVIRONMENT_ID_PATTERN.fullmatch(resolved_id) or not (
        _environment_matches(reference, resolved_id)
    ):
        raise RuntimeError(
            "Azure ML registry operation scope could not produce the requested "
            f"immutable environment resource ID. requested={reference!r}, "
            f"resolved={resolved_id!r}"
        )

    environment_id = str(getattr(environment, "id", "") or "").rstrip("/")
    returned_reference = _parse_registry_environment_reference(environment_id)
    if (
        not environment_id
        or not returned_reference
        or (
            returned_reference[2].lower() != registry_name.lower()
            or returned_reference[3].lower() != environment_name.lower()
            or returned_reference[4] != version
        )
    ):
        raise RuntimeError(
            "Azure ML returned a missing or mismatched ID for the exact registry "
            f"environment lookup. requested={reference!r}, "
            f"returned={environment_id!r}"
        )
    if FULL_ENVIRONMENT_ID_PATTERN.fullmatch(environment_id) and (
        environment_id.lower() != resolved_id.lower()
    ):
        raise RuntimeError(
            "Azure ML returned a full registry environment ID that conflicts with "
            f"its authoritative operation scope. returned={environment_id!r}, "
            f"scope={resolved_id!r}"
        )
    return (
        registry_client,
        environment,
        registry_name,
        environment_name,
        version,
        resolved_id,
    )


def _workspace_environment_name(registry_name: str, environment_name: str) -> str:
    raw_name = f"registry-{registry_name}-{environment_name}".lower()
    safe_name = re.sub(r"[^a-z0-9-]+", "-", raw_name).strip("-")
    safe_name = re.sub(r"-+", "-", safe_name)
    if not safe_name:
        raise RuntimeError("Registry environment produced an empty workspace name.")
    if len(safe_name) <= 255:
        return safe_name
    suffix = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
    return f"{safe_name[:242].rstrip('-')}-{suffix}"


def _registry_provenance(
    registry_name: str,
    environment_name: str,
    version: str,
    reference: str,
    manifest: str,
) -> dict[str, str]:
    return {
        SOURCE_REGISTRY_PROPERTY: registry_name,
        SOURCE_ENVIRONMENT_PROPERTY: environment_name,
        SOURCE_VERSION_PROPERTY: version,
        SOURCE_REFERENCE_PROPERTY: reference,
        SOURCE_MANIFEST_PROPERTY: manifest,
    }


def _validate_sdk_registry_helper() -> None:
    try:
        installed_version = package_version("azure-ai-ml")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "azure-ai-ml is not installed; version "
            f"{SUPPORTED_AZURE_AI_ML_VERSION} is required for registry import."
        ) from exc
    if installed_version != SUPPORTED_AZURE_AI_ML_VERSION:
        raise RuntimeError(
            "Registry environment import requires the validated internal "
            f"azure-ai-ml surface from version {SUPPORTED_AZURE_AI_ML_VERSION}; "
            f"installed version is {installed_version}."
        )
    if not callable(get_storage_details_for_registry_assets):
        raise RuntimeError(
            "azure-ai-ml registry storage helper is unavailable. Install the "
            f"pinned azure-ai-ml=={SUPPORTED_AZURE_AI_ML_VERSION} dependency."
        )


def _safe_source_prefix(source_uri: str) -> tuple[str, str, str]:
    parsed = urlsplit(source_uri)
    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or len(path_parts) < 2
    ):
        raise RuntimeError(
            "Registry build path must be an HTTPS blob container prefix without "
            "query parameters."
        )
    container, *prefix_parts = path_parts
    if not STORAGE_CONTAINER_PATTERN.fullmatch(container):
        raise RuntimeError("Registry build path contains an unsafe container name.")
    prefix_path = PurePosixPath(*prefix_parts)
    if (
        prefix_path.is_absolute()
        or ".." in prefix_path.parts
        or any(not part or part in {".", ".."} for part in prefix_path.parts)
    ):
        raise RuntimeError("Registry build path contains an unsafe source prefix.")
    container_url = urlunsplit((parsed.scheme, parsed.netloc, f"/{container}", "", ""))
    return container_url, prefix_path.as_posix().rstrip("/") + "/", parsed.netloc


def _authenticated_container_url(
    registry_client: object,
    registry_name: str,
    environment_name: str,
    version: str,
    source_uri: str,
) -> tuple[str, str]:
    _validate_sdk_registry_helper()
    service_client = getattr(registry_client.environments, "_service_client", None)
    operation_scope = getattr(registry_client.environments, "_operation_scope", None)
    resource_group_name = str(getattr(operation_scope, "resource_group_name", "") or "")
    if service_client is None or not resource_group_name:
        raise RuntimeError(
            "azure-ai-ml registry environment internals are unavailable; the "
            "service client and operation scope are required for authenticated "
            "build-context import."
        )
    try:
        storage_uri, credential_type = get_storage_details_for_registry_assets(
            service_client=service_client,
            asset_type=REGISTRY_ASSET_TYPE,
            asset_name=environment_name,
            asset_version=version,
            rg_name=resource_group_name,
            reg_name=registry_name,
            uri=source_uri,
        )
    except Exception:
        raise RuntimeError(
            "Azure ML could not issue authenticated registry storage access for "
            f"{registry_name}/{environment_name}:{version}."
        ) from None
    if credential_type != "SAS":
        raise RuntimeError(
            "Registry build-context import requires container-scoped SAS access."
        )

    source_container_url, prefix, source_host = _safe_source_prefix(source_uri)
    parsed_sas = urlsplit(str(storage_uri))
    if (
        parsed_sas.scheme.lower() != "https"
        or parsed_sas.netloc.lower() != source_host.lower()
        or not parsed_sas.query
        or parsed_sas.fragment
    ):
        raise RuntimeError(
            "Azure ML returned invalid authenticated registry storage details."
        )
    if parsed_sas.path.rstrip("/") != urlsplit(source_container_url).path.rstrip("/"):
        raise RuntimeError(
            "Azure ML returned registry storage access for an unexpected container."
        )
    return urlunsplit(parsed_sas), prefix


def _open_url(request: Request):
    return urlopen(request, timeout=60)


def _request_bytes(url: str, maximum_bytes: int, purpose: str) -> tuple[bytes, object]:
    try:
        with _open_url(Request(url, headers={"Accept": "application/xml"})) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Registry {purpose} returned an invalid Content-Length."
                    ) from exc
                if declared_length < 0 or declared_length > maximum_bytes:
                    raise RuntimeError(
                        f"Registry {purpose} exceeds the allowed size bound."
                    )
            body = response.read(maximum_bytes + 1)
            if len(body) > maximum_bytes:
                raise RuntimeError(
                    f"Registry {purpose} exceeds the allowed size bound."
                )
            if content_length is not None and len(body) != declared_length:
                raise RuntimeError(
                    f"Registry {purpose} Content-Length did not match the response."
                )
            return body, response.headers
    except (HTTPError, URLError, TimeoutError):
        raise RuntimeError(f"Registry {purpose} request failed.") from None


def _url_with_query(url: str, additions: dict[str, str]) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(additions)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))


def _safe_blob_relative_path(blob_name: str, prefix: str) -> Path:
    decoded_name = unquote(blob_name)
    if decoded_name != blob_name or not decoded_name.startswith(prefix):
        raise RuntimeError("Registry listing returned a blob outside the build prefix.")
    relative_name = decoded_name[len(prefix) :]
    relative_path = PurePosixPath(relative_name)
    if (
        not relative_name
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or "\\" in relative_name
        or any(ord(character) < 32 for character in relative_name)
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise RuntimeError("Registry listing returned an unsafe blob path.")
    return Path(*relative_path.parts)


def _listing_entries(listing: bytes, prefix: str) -> list[tuple[str, Path, int, str]]:
    try:
        root = ET.fromstring(listing)
    except ET.ParseError as exc:
        raise RuntimeError("Registry build-context listing is not valid XML.") from exc
    if (root.findtext(".//NextMarker") or "").strip():
        raise RuntimeError(
            "Registry build-context listing requires unsupported pagination."
        )
    entries = []
    seen_paths = set()
    total_size = 0
    for blob in root.findall(".//Blob"):
        name = blob.findtext("Name") or ""
        relative_path = _safe_blob_relative_path(name, prefix)
        if relative_path in seen_paths:
            raise RuntimeError("Registry listing contains a duplicate blob path.")
        seen_paths.add(relative_path)
        length_text = blob.findtext("./Properties/Content-Length")
        md5_text = blob.findtext("./Properties/Content-MD5")
        try:
            length = int(length_text or "")
            expected_md5 = base64.b64decode(md5_text or "", validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                "Registry listing contains invalid blob length or MD5 metadata."
            ) from exc
        if length < 0 or length > MAX_REGISTRY_BUILD_FILE_BYTES:
            raise RuntimeError("Registry build-context file exceeds the size bound.")
        if len(expected_md5) != 16:
            raise RuntimeError("Registry listing contains invalid blob MD5 metadata.")
        total_size += length
        if total_size > MAX_REGISTRY_BUILD_TOTAL_BYTES:
            raise RuntimeError("Registry build context exceeds the total size bound.")
        entries.append((name, relative_path, length, md5_text or ""))
        if len(entries) > MAX_REGISTRY_BUILD_FILES:
            raise RuntimeError("Registry build context exceeds the file-count bound.")
    if not entries:
        raise RuntimeError("Registry build context did not contain any files.")
    entries.sort(key=lambda item: item[0])
    return entries


def _download_registry_build_context(
    registry_client: object,
    registry_name: str,
    environment_name: str,
    version: str,
    source_uri: str,
    destination: Path,
) -> tuple[str, set[str]]:
    container_sas_url, prefix = _authenticated_container_url(
        registry_client,
        registry_name,
        environment_name,
        version,
        source_uri,
    )
    listing_url = _url_with_query(
        container_sas_url,
        {"restype": "container", "comp": "list", "prefix": prefix},
    )
    listing, _ = _request_bytes(
        listing_url,
        MAX_REGISTRY_LISTING_BYTES,
        "build-context listing",
    )
    entries = _listing_entries(listing, prefix)
    manifest_entries = []
    downloaded_paths = set()
    parsed_container = urlsplit(container_sas_url)
    for blob_name, relative_path, expected_length, expected_md5_text in entries:
        blob_url = urlunsplit(
            (
                parsed_container.scheme,
                parsed_container.netloc,
                f"{parsed_container.path.rstrip('/')}/{quote(blob_name, safe='/')}",
                parsed_container.query,
                "",
            )
        )
        body, headers = _request_bytes(
            blob_url,
            MAX_REGISTRY_BUILD_FILE_BYTES,
            "build-context download",
        )
        if len(body) != expected_length:
            raise RuntimeError(
                "Registry build-context blob length did not match its listing."
            )
        response_md5 = headers.get("Content-MD5")
        if response_md5 != expected_md5_text:
            raise RuntimeError(
                "Registry build-context blob MD5 did not match its listing."
            )
        actual_md5 = base64.b64encode(
            hashlib.md5(body, usedforsecurity=False).digest()
        ).decode("ascii")
        if actual_md5 != expected_md5_text:
            raise RuntimeError("Registry build-context blob failed MD5 validation.")
        output_path = destination / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(body)
        relative_posix = relative_path.as_posix()
        downloaded_paths.add(relative_posix)
        manifest_entries.append(
            {
                "path": relative_posix,
                "sha256": hashlib.sha256(body).hexdigest(),
                "size": len(body),
            }
        )
    manifest = hashlib.sha256(
        json.dumps(
            manifest_entries,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return manifest, downloaded_paths


def _environment_properties(environment: object) -> dict[str, str]:
    properties = getattr(environment, "properties", None)
    return dict(properties) if isinstance(properties, dict) else {}


def _environment_provenance(environment: object) -> dict[str, str]:
    tags = getattr(environment, "tags", None)
    metadata = (
        _environment_properties(environment),
        dict(tags) if isinstance(tags, dict) else {},
    )
    provenance = {}
    for key in SOURCE_PROVENANCE_PROPERTIES:
        values = [source[key] for source in metadata if key in source]
        if values and any(value != values[0] for value in values[1:]):
            raise RuntimeError(
                "Workspace environment contains conflicting registry provenance."
            )
        if values:
            provenance[key] = values[0]
    return provenance


def _validate_workspace_environment(
    environment: object,
    expected_name: str,
    expected_version: str,
    expected_provenance: dict[str, str],
) -> object:
    actual_name = str(getattr(environment, "name", "") or "")
    actual_version = str(getattr(environment, "version", "") or "")
    try:
        actual_provenance = _environment_provenance(environment)
    except RuntimeError:
        actual_provenance = {}
    if (
        actual_name != expected_name
        or actual_version != expected_version
        or any(
            actual_provenance.get(key) != value
            for key, value in expected_provenance.items()
        )
    ):
        raise RuntimeError(
            "Workspace environment name/version collides with a different "
            "registry source or manifest."
        )
    environment_id = str(getattr(environment, "id", "") or "")
    if not environment_id or not _environment_matches(
        f"azureml:{expected_name}:{expected_version}",
        environment_id,
    ):
        raise RuntimeError(
            "Azure ML returned a missing or mismatched workspace environment ID."
        )
    return environment


def _existing_workspace_environment(
    ml_client: object,
    name: str,
    version: str,
) -> object | None:
    try:
        return ml_client.environments.get(name=name, version=version)
    except ResourceNotFoundError:
        return None


def _materialize_build_environment(
    ml_client: object,
    registry_client: object,
    registry_environment: object,
    registry_name: str,
    environment_name: str,
    version: str,
    reference: str,
    workspace_name: str,
) -> object:
    build = getattr(registry_environment, "build", None)
    source_uri = str(getattr(build, "path", "") or "")
    dockerfile_path = str(getattr(build, "dockerfile_path", "") or "")
    dockerfile = PurePosixPath(dockerfile_path)
    if (
        not source_uri
        or not dockerfile_path
        or dockerfile.is_absolute()
        or ".." in dockerfile.parts
    ):
        raise RuntimeError(
            "Registry build-backed environment has an unsafe or incomplete build "
            "source."
        )
    with TemporaryDirectory(prefix="azureml-registry-environment-") as temp_directory:
        local_path = Path(temp_directory)
        manifest, downloaded_paths = _download_registry_build_context(
            registry_client,
            registry_name,
            environment_name,
            version,
            source_uri,
            local_path,
        )
        normalized_dockerfile = dockerfile.as_posix()
        if normalized_dockerfile not in downloaded_paths:
            raise RuntimeError(
                "Registry build context does not contain its declared Dockerfile."
            )
        provenance = _registry_provenance(
            registry_name,
            environment_name,
            version,
            reference,
            manifest,
        )
        existing = _existing_workspace_environment(
            ml_client,
            workspace_name,
            version,
        )
        if existing is not None:
            validated = _validate_workspace_environment(
                existing,
                workspace_name,
                version,
                provenance,
            )
            if getattr(validated, "image", None) or not getattr(
                validated,
                "build",
                None,
            ):
                raise RuntimeError(
                    "Workspace environment source conflicts with the registry "
                    "build context."
                )
            return validated
        environment = Environment(
            name=workspace_name,
            version=version,
            build=BuildContext(
                path=str(local_path),
                dockerfile_path=normalized_dockerfile,
            ),
            tags=provenance,
            properties=provenance,
        )
        created = ml_client.environments.create_or_update(environment)
        validated = _validate_workspace_environment(
            created,
            workspace_name,
            version,
            provenance,
        )
        if getattr(validated, "image", None) or not getattr(
            validated,
            "build",
            None,
        ):
            raise RuntimeError(
                "Azure ML created a workspace environment with an unexpected "
                "source shape."
            )
        return validated


def _materialize_image_environment(
    ml_client: object,
    registry_environment: object,
    registry_name: str,
    environment_name: str,
    version: str,
    reference: str,
    workspace_name: str,
) -> object:
    image = str(getattr(registry_environment, "image", "") or "")
    if not IMMUTABLE_IMAGE_PATTERN.fullmatch(image):
        raise RuntimeError(
            "Registry image-backed environment must use an immutable sha256 digest."
        )
    manifest = hashlib.sha256(image.encode("utf-8")).hexdigest()
    provenance = _registry_provenance(
        registry_name,
        environment_name,
        version,
        reference,
        manifest,
    )
    existing = _existing_workspace_environment(ml_client, workspace_name, version)
    if existing is not None:
        validated = _validate_workspace_environment(
            existing,
            workspace_name,
            version,
            provenance,
        )
        if str(getattr(validated, "image", "") or "") != image:
            raise RuntimeError(
                "Workspace environment image conflicts with the immutable registry "
                "source."
            )
        return validated
    environment = Environment(
        name=workspace_name,
        version=version,
        image=image,
        tags=provenance,
        properties=provenance,
    )
    created = ml_client.environments.create_or_update(environment)
    validated = _validate_workspace_environment(
        created,
        workspace_name,
        version,
        provenance,
    )
    if str(getattr(validated, "image", "") or "") != image:
        raise RuntimeError(
            "Azure ML created a workspace environment with an unexpected image."
        )
    return validated


def resolve_batch_environment(ml_client: object, reference: str) -> str | object:
    if not _parse_registry_environment_reference(reference):
        return reference
    (
        registry_client,
        registry_environment,
        registry_name,
        environment_name,
        version,
        _,
    ) = _get_registry_environment(ml_client, reference)
    workspace_name = _workspace_environment_name(
        registry_name,
        environment_name,
    )
    build = getattr(registry_environment, "build", None)
    image = getattr(registry_environment, "image", None)
    conda_file = getattr(registry_environment, "conda_file", None)
    if build and (image or conda_file):
        raise RuntimeError(
            "Registry environment has conflicting build, image, or Conda sources."
        )
    if image and conda_file:
        raise RuntimeError(
            "Registry environment has conflicting image and Conda sources."
        )
    if build:
        return _materialize_build_environment(
            ml_client,
            registry_client,
            registry_environment,
            registry_name,
            environment_name,
            version,
            reference,
            workspace_name,
        )
    if image:
        return _materialize_image_environment(
            ml_client,
            registry_environment,
            registry_name,
            environment_name,
            version,
            reference,
            workspace_name,
        )
    raise RuntimeError(
        "Registry environment source is unsupported; expected an immutable image "
        "or a downloadable build context."
    )


def resolve_scoring_code(
    repository_root: str,
    scoring_code_directory: str,
    scoring_script: str,
) -> tuple[Path, str]:
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Consumer repository root does not exist: {root}")

    code_input = Path(scoring_code_directory)
    script_input = Path(scoring_script)
    if code_input.is_absolute() or ".." in code_input.parts:
        raise ValueError(
            "scoring_code_directory must be a traversal-free path relative to "
            "the checked-out consumer repository"
        )
    if script_input.is_absolute() or ".." in script_input.parts:
        raise ValueError(
            "scoring_script must be a traversal-free path relative to "
            "scoring_code_directory"
        )
    if not code_input.parts or code_input == Path("."):
        raise ValueError(
            "scoring_code_directory must name a dedicated consumer scoring directory"
        )
    if code_input.parts[0] == ".mlops-python-sdk":
        raise ValueError(
            "scoring_code_directory must come from the consumer repository, not "
            "the checked-out reusable SDK layer"
        )

    code_directory = (root / code_input).resolve()
    try:
        code_directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "scoring_code_directory resolves outside the checked-out consumer "
            "repository"
        ) from exc
    if not code_directory.is_dir():
        raise ValueError(f"Scoring code directory does not exist: {code_directory}")

    script_path = (code_directory / script_input).resolve()
    try:
        relative_script = script_path.relative_to(code_directory)
    except ValueError as exc:
        raise ValueError(
            "scoring_script resolves outside scoring_code_directory"
        ) from exc
    if not script_path.is_file():
        raise ValueError(f"Scoring script does not exist: {script_path}")
    return code_directory, relative_script.as_posix()


def _environment_matches(requested: str, actual: str) -> bool:
    requested_value = requested.rstrip("/")
    actual_value = actual.rstrip("/")
    if requested_value.lower() == actual_value.lower():
        return True

    registry_match = REGISTRY_ENVIRONMENT_PATTERN.fullmatch(requested_value)
    if registry_match:
        registry_name, environment_name, version = registry_match.groups()
        expected_suffix = (
            f"/registries/{registry_name}/environments/{environment_name}/"
            f"versions/{version}"
        )
        return actual_value.lower().endswith(expected_suffix.lower())

    workspace_match = re.fullmatch(
        r"azureml:([^:\s]+):(\d+)",
        requested_value,
        re.IGNORECASE,
    )
    if workspace_match:
        environment_name, version = workspace_match.groups()
        expected_suffix = f"/environments/{environment_name}/versions/{version}"
        return actual_value.lower().endswith(expected_suffix.lower())
    return False


def verify_live_deployment(
    deployment: object,
    workspace_environment: str | object,
    requested_source: str,
    scoring_script: str,
) -> None:
    expected_environment_id = str(
        getattr(workspace_environment, "id", workspace_environment) or ""
    )
    live_environment = getattr(deployment, "environment", None)
    if not live_environment or not _environment_matches(
        expected_environment_id,
        str(live_environment),
    ):
        raise RuntimeError(
            "Azure ML batch deployment did not persist the requested immutable "
            f"environment in the workspace. requested={expected_environment_id!r}, "
            f"live={live_environment!r}"
        )
    registry_reference = _parse_registry_environment_reference(requested_source)
    if registry_reference:
        _, _, registry_name, environment_name, version = registry_reference
        provenance = _environment_provenance(workspace_environment)
        manifest = provenance.get(SOURCE_MANIFEST_PROPERTY, "")
        if not re.fullmatch(r"[0-9a-f]{64}", manifest):
            raise RuntimeError(
                "Workspace environment does not retain a valid source manifest."
            )
        expected_source = {
            SOURCE_REGISTRY_PROPERTY: registry_name,
            SOURCE_ENVIRONMENT_PROPERTY: environment_name,
            SOURCE_VERSION_PROPERTY: version,
            SOURCE_REFERENCE_PROPERTY: requested_source,
            SOURCE_MANIFEST_PROPERTY: manifest,
        }
        if any(provenance.get(key) != value for key, value in expected_source.items()):
            raise RuntimeError(
                "Workspace environment does not retain the requested registry "
                "source provenance."
            )

    live_code_configuration = getattr(deployment, "code_configuration", None)
    live_code = getattr(live_code_configuration, "code", None)
    live_scoring_script = getattr(
        live_code_configuration,
        "scoring_script",
        None,
    )
    if not live_code_configuration or not live_code:
        raise RuntimeError(
            "Azure ML batch deployment did not persist a code configuration; "
            "refusing to invoke a deployment that could synthesize an anonymous "
            "environment build"
        )
    if Path(str(live_scoring_script or "")).as_posix() != scoring_script:
        raise RuntimeError(
            "Azure ML batch deployment persisted an unexpected scoring script. "
            f"requested={scoring_script!r}, live={live_scoring_script!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update a batch deployment and make it the default."
    )
    add_workspace_arguments(parser)
    parser.add_argument("--deployment_name", required=True)
    parser.add_argument("--description")
    parser.add_argument("--endpoint_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--compute", required=True)
    parser.add_argument(
        "--environment",
        type=validate_immutable_environment_reference,
        default=DEFAULT_BATCH_ENVIRONMENT,
        help="Immutable versioned Azure ML environment reference.",
    )
    parser.add_argument(
        "--repository_root",
        required=True,
        help="Root of the checked-out consumer repository.",
    )
    parser.add_argument(
        "--scoring_code_directory",
        required=True,
        help="Consumer-repository-relative scoring code directory.",
    )
    parser.add_argument(
        "--scoring_script",
        required=True,
        help="Scoring script path relative to the scoring code directory.",
    )
    parser.add_argument("--instance_count", type=int, default=2)
    parser.add_argument("--max_concurrency_per_instance", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--output_file_name", default="predictions.csv")
    return parser.parse_args()


def run(args: argparse.Namespace):
    requested_environment = validate_immutable_environment_reference(args.environment)
    code_directory, scoring_script = resolve_scoring_code(
        args.repository_root,
        args.scoring_code_directory,
        args.scoring_script,
    )
    ml_client = create_ml_client(args)
    model = get_registered_model(
        ml_client,
        args.model_name,
        args.model_version,
        require_mlflow=True,
    )
    environment = resolve_batch_environment(ml_client, requested_environment)
    code_configuration = CodeConfiguration(
        code=str(code_directory),
        scoring_script=scoring_script,
    )
    deployment = BatchDeployment(
        name=args.deployment_name,
        description=args.description,
        endpoint_name=args.endpoint_name,
        model=model.id,
        environment=environment,
        code_configuration=code_configuration,
        compute=args.compute,
        instance_count=args.instance_count,
        max_concurrency_per_instance=args.max_concurrency_per_instance,
        mini_batch_size=args.mini_batch_size,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name=args.output_file_name,
    )
    live_deployment = wait_for_resource_create_or_update(
        lambda: ml_client.batch_deployments.begin_create_or_update(deployment),
        lambda: ml_client.batch_deployments.get(
            args.deployment_name,
            endpoint_name=args.endpoint_name,
        ),
        f"batch deployment {args.deployment_name}",
    )
    verify_live_deployment(
        live_deployment,
        environment,
        requested_environment,
        scoring_script,
    )

    endpoint = ml_client.batch_endpoints.get(args.endpoint_name)
    endpoint.defaults.deployment_name = args.deployment_name
    return wait_for_resource_create_or_update(
        lambda: ml_client.batch_endpoints.begin_create_or_update(endpoint),
        lambda: ml_client.batch_endpoints.get(args.endpoint_name),
        f"batch endpoint {args.endpoint_name}",
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
