# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import re
import ssl
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from azure.ai.ml import MLClient
from azure.ai.ml.entities import IdentityConfiguration, ManagedIdentityConfiguration
from azure.core.exceptions import ResourceExistsError
from azure.identity import AzureCliCredential, DefaultAzureCredential

AZUREML_CREDENTIAL_MODE = "AZUREML_CREDENTIAL_MODE"
AZURE_CLI_CREDENTIAL_MODE = "azure-cli"
DEFAULT_CREDENTIAL_MODE = "default"
MANAGEMENT_SCOPE = "https://management.azure.com/.default"
SUCCESS_JOB_STATUSES = {"Completed"}
FAILED_JOB_STATUSES = {"Failed", "Canceled", "Cancelled", "NotResponding"}
TERMINAL_JOB_STATUSES = {
    "Completed",
    "Failed",
    "Canceled",
    "Cancelled",
    "NotResponding",
    "Paused",
}
SUCCESS_PROVISIONING_STATES = {"Succeeded"}
FAILED_PROVISIONING_STATES = {"Failed", "Canceled", "Cancelled"}
DEFAULT_RESOURCE_POLL_INTERVAL_SECONDS = 5
DEFAULT_RESOURCE_MAX_POLLS = 120
DEFAULT_RESOURCE_UPDATE_ATTEMPTS = 3
DIAGNOSTICS_DIRECTORY = "aml-diagnostics"
MAX_DIAGNOSTIC_FILES = 20
MAX_DIAGNOSTIC_FILE_BYTES = 256_000
MAX_DIAGNOSTIC_OUTPUT_CHARS = 20_000
MAX_DIAGNOSTIC_LINE_CHARS = 2_000
DIAGNOSTIC_CONTEXT_LINES = 4
ERROR_LINE_PATTERN = re.compile(
    r"traceback|exception|error|failed|fatal|stderr|exit code",
    re.IGNORECASE,
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b("
    r"access[_-]?token|api[_-]?key|authorization|client[_-]?secret|credential|"
    r"password|private[_-]?key|secret|token"
    r")(\s*[:=]\s*)([^\r\n,;]+)"
)
BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
SAS_PARAMETER_PATTERN = re.compile(
    r"(?i)([?&](?:sig|se|sp|spr|sr|sv|st)=)[^&\s]+"
)
IMAGE_DIGEST_PATTERN = re.compile(r"@sha256:[0-9a-f]{64}$", re.IGNORECASE)
USER_ASSIGNED_IDENTITY_RESOURCE_ID_PATTERN = re.compile(
    r"^/subscriptions/[^/]+/resourceGroups/[^/]+/providers/"
    r"Microsoft\.ManagedIdentity/userAssignedIdentities/[^/]+$",
    re.IGNORECASE,
)
CA_BUNDLE_ENVIRONMENT_VARIABLES = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")


def add_workspace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)


def create_azure_credential() -> Any:
    is_ci = _is_truthy(os.environ.get("GITHUB_ACTIONS")) or _is_truthy(
        os.environ.get("CI")
    )
    mode = os.environ.get(AZUREML_CREDENTIAL_MODE)
    if mode is None:
        mode = AZURE_CLI_CREDENTIAL_MODE if is_ci else DEFAULT_CREDENTIAL_MODE
    mode = mode.strip().lower()

    if mode not in {AZURE_CLI_CREDENTIAL_MODE, DEFAULT_CREDENTIAL_MODE}:
        raise RuntimeError(
            f"Unsupported {AZUREML_CREDENTIAL_MODE} value '{mode}'. "
            f"Use '{AZURE_CLI_CREDENTIAL_MODE}' or '{DEFAULT_CREDENTIAL_MODE}'."
        )
    if is_ci and mode != AZURE_CLI_CREDENTIAL_MODE:
        raise RuntimeError(
            f"{AZUREML_CREDENTIAL_MODE} must be '{AZURE_CLI_CREDENTIAL_MODE}' in CI. "
            "Run azure/login before the Python SDK step; managed identity fallback "
            "is intentionally disabled."
        )

    if mode == AZURE_CLI_CREDENTIAL_MODE:
        credential = AzureCliCredential()
        credential_description = "Azure CLI credential created by azure/login"
    else:
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=True,
            exclude_managed_identity_credential=True,
        )
        credential_description = "local DefaultAzureCredential"

    try:
        credential.get_token(MANAGEMENT_SCOPE)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to acquire an Azure management token with "
            f"{credential_description}. "
            + (
                "Verify that azure/login completed successfully in this job."
                if mode == AZURE_CLI_CREDENTIAL_MODE
                else "Sign in with Azure CLI or configure a supported local credential."
            )
        ) from exc
    return credential


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes"}


def create_ml_client(args: argparse.Namespace) -> MLClient:
    credential = create_azure_credential()
    return MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )


@contextmanager
def use_private_ca_bundle(ca_bundle: str) -> Iterator[Path]:
    if not ca_bundle or not ca_bundle.strip():
        raise RuntimeError(
            "Private endpoint invocation requires a CA bundle fetched from the "
            "configured Key Vault secret."
        )

    ca_bundle_path = Path(ca_bundle).expanduser()
    if not ca_bundle_path.is_file():
        raise RuntimeError(
            f"Private endpoint CA bundle '{ca_bundle_path}' is missing or unreadable."
        )

    try:
        ssl.create_default_context(cafile=str(ca_bundle_path))
    except (OSError, ssl.SSLError) as exc:
        raise RuntimeError(
            f"Private endpoint CA bundle '{ca_bundle_path}' is not a valid PEM "
            "certificate bundle."
        ) from exc

    previous_values = {
        variable: os.environ.get(variable)
        for variable in CA_BUNDLE_ENVIRONMENT_VARIABLES
    }
    resolved_path = ca_bundle_path.resolve()
    for variable in CA_BUNDLE_ENVIRONMENT_VARIABLES:
        os.environ[variable] = str(resolved_path)

    try:
        yield resolved_path
    finally:
        for variable, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous_value


def get_user_assigned_identity_configuration(
    resource_id: str | None,
) -> IdentityConfiguration | None:
    if resource_id is None or not resource_id.strip():
        return None
    if (
        resource_id != resource_id.strip()
        or not USER_ASSIGNED_IDENTITY_RESOURCE_ID_PATTERN.fullmatch(resource_id)
    ):
        raise RuntimeError(
            "Endpoint user-assigned identity must be a full Azure resource ID in "
            "'/subscriptions/<subscription>/resourceGroups/<resource-group>/providers/"
            "Microsoft.ManagedIdentity/userAssignedIdentities/<identity>' format. "
            "System-assigned and AKS node identity fallback are not supported."
        )
    return IdentityConfiguration(
        type="user_assigned",
        user_assigned_identities=[
            ManagedIdentityConfiguration(resource_id=resource_id)
        ],
    )


def wait_for_poller(poller: Any) -> Any:
    return poller.result()


def wait_for_resource_create_or_update(
    begin_create_or_update: Callable[[], Any],
    get_resource: Callable[[], Any],
    resource_description: str,
    *,
    poll_interval_seconds: int = DEFAULT_RESOURCE_POLL_INTERVAL_SECONDS,
    max_state_polls: int = DEFAULT_RESOURCE_MAX_POLLS,
    max_update_attempts: int = DEFAULT_RESOURCE_UPDATE_ATTEMPTS,
) -> Any:
    if max_state_polls < 1:
        raise ValueError("max_state_polls must be at least 1")
    if max_update_attempts < 1:
        raise ValueError("max_update_attempts must be at least 1")

    for attempt in range(1, max_update_attempts + 1):
        try:
            wait_for_poller(begin_create_or_update())
        except ResourceExistsError as exc:
            if attempt == max_update_attempts:
                raise RuntimeError(
                    f"Azure ML {resource_description} update remained blocked by "
                    f"another operation after {max_update_attempts} attempts"
                ) from exc
            print(
                f"Azure ML {resource_description} already has an operation in "
                f"progress; waiting for it before retrying update "
                f"({attempt}/{max_update_attempts})",
                flush=True,
            )
            wait_for_resource_terminal_state(
                get_resource,
                resource_description,
                poll_interval_seconds=poll_interval_seconds,
                max_state_polls=max_state_polls,
            )
            continue

        return wait_for_resource_terminal_state(
            get_resource,
            resource_description,
            poll_interval_seconds=poll_interval_seconds,
            max_state_polls=max_state_polls,
        )

    raise AssertionError("unreachable")


def wait_for_resource_terminal_state(
    get_resource: Callable[[], Any],
    resource_description: str,
    *,
    poll_interval_seconds: int = DEFAULT_RESOURCE_POLL_INTERVAL_SECONDS,
    max_state_polls: int = DEFAULT_RESOURCE_MAX_POLLS,
) -> Any:
    if max_state_polls < 1:
        raise ValueError("max_state_polls must be at least 1")

    resource = None
    for poll_number in range(1, max_state_polls + 1):
        if resource is None:
            resource = get_resource()
        state = str(getattr(resource, "provisioning_state", None))
        print(
            f"Azure ML {resource_description} provisioning state: {state}",
            flush=True,
        )
        if state in SUCCESS_PROVISIONING_STATES:
            return resource
        if state in FAILED_PROVISIONING_STATES:
            raise RuntimeError(
                f"Azure ML {resource_description} provisioning finished with "
                f"state {state}"
            )
        if poll_number == max_state_polls:
            break
        time.sleep(poll_interval_seconds)
        resource = get_resource()

    raise TimeoutError(
        f"Azure ML {resource_description} did not reach a terminal provisioning "
        f"state after {max_state_polls} state checks"
    )


def wait_for_job(
    ml_client: MLClient,
    job_name: str,
    poll_interval_seconds: int = 15,
) -> Any:
    while True:
        job = ml_client.jobs.get(job_name)
        status = str(job.status)
        print(f"Job {job_name} status: {status}", flush=True)
        if status in TERMINAL_JOB_STATUSES:
            break
        time.sleep(poll_interval_seconds)

    if status not in SUCCESS_JOB_STATUSES:
        print_job_diagnostics(ml_client, job)
        raise RuntimeError(f"Azure ML job {job_name} finished with status {status}")
    return job


def print_job_diagnostics(ml_client: MLClient, job: Any) -> None:
    error = getattr(job, "error", None)
    if error:
        print(f"Job error: {_bounded_redacted_text(str(error))}", flush=True)

    try:
        children = list(ml_client.jobs.list(parent_job_name=job.name))
    except Exception as exc:
        print(
            "Unable to list child-job diagnostics: "
            f"{_bounded_redacted_text(str(exc))}",
            flush=True,
        )
        children = []

    failed_children = []
    for child in children:
        child_error = getattr(child, "error", None)
        print(
            f"Child job {child.name}: status={child.status}"
            + (
                f", error={_bounded_redacted_text(str(child_error))}"
                if child_error
                else ""
            ),
            flush=True,
        )
        if str(child.status) in FAILED_JOB_STATUSES:
            failed_children.append(child)

    diagnostic_jobs = failed_children or [job]
    diagnostics_root = Path(
        os.environ.get("AML_DIAGNOSTICS_DIR", DIAGNOSTICS_DIRECTORY)
    )
    parent_directory = diagnostics_root / _safe_path_component(str(job.name))
    print(f"AML diagnostics directory: {parent_directory}", flush=True)
    for diagnostic_job in diagnostic_jobs:
        _download_and_print_job_diagnostics(
            ml_client,
            diagnostic_job,
            parent_directory,
        )


def _download_and_print_job_diagnostics(
    ml_client: MLClient,
    job: Any,
    parent_directory: Path,
) -> None:
    job_name = str(job.name)
    download_directory = parent_directory / _safe_path_component(job_name)
    download_directory.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading diagnostics for job {job_name} to {download_directory}",
        flush=True,
    )
    try:
        ml_client.jobs.download(
            name=job_name,
            download_path=str(download_directory),
            all=False,
        )
    except Exception as exc:
        print(
            f"Unable to download diagnostics for job {job_name}: "
            f"{_bounded_redacted_text(str(exc))}",
            flush=True,
        )
        return

    files = sorted(path for path in download_directory.rglob("*") if path.is_file())
    if not _contains_log_file(files):
        print(
            f"No log files found in standard diagnostics for job {job_name}; "
            "retrying full job download",
            flush=True,
        )
        try:
            ml_client.jobs.download(
                name=job_name,
                download_path=str(download_directory),
                all=True,
            )
        except Exception as exc:
            print(
                f"Unable to download full diagnostics for job {job_name}: "
                f"{_bounded_redacted_text(str(exc))}",
                flush=True,
            )
        files = sorted(
            path for path in download_directory.rglob("*") if path.is_file()
        )

    if not files:
        print(f"No diagnostic files downloaded for job {job_name}", flush=True)
        return

    remaining_chars = MAX_DIAGNOSTIC_OUTPUT_CHARS
    for path in files[:MAX_DIAGNOSTIC_FILES]:
        if remaining_chars <= 0:
            break
        try:
            with path.open("rb") as diagnostic_file:
                content = diagnostic_file.read(MAX_DIAGNOSTIC_FILE_BYTES).decode(
                    "utf-8",
                    errors="replace",
                )
        except Exception as exc:
            print(
                f"Unable to read diagnostic file {path}: "
                f"{_bounded_redacted_text(str(exc))}",
                flush=True,
            )
            continue

        excerpt = _diagnostic_excerpt(content, remaining_chars)
        if not excerpt:
            continue
        relative_path = path.relative_to(download_directory)
        print(f"--- Diagnostics from {job_name}/{relative_path} ---", flush=True)
        print(excerpt, flush=True)
        remaining_chars -= len(excerpt)

    if len(files) > MAX_DIAGNOSTIC_FILES:
        print(
            f"Diagnostic file output limited to {MAX_DIAGNOSTIC_FILES} of "
            f"{len(files)} files for job {job_name}",
            flush=True,
        )
    if remaining_chars <= 0:
        print(
            f"Diagnostic output limited to {MAX_DIAGNOSTIC_OUTPUT_CHARS} "
            f"characters for job {job_name}",
            flush=True,
        )


def _diagnostic_excerpt(content: str, max_chars: int) -> str:
    lines = content.splitlines()
    selected_indexes = set()
    for index, line in enumerate(lines):
        if ERROR_LINE_PATTERN.search(line):
            start = max(0, index - DIAGNOSTIC_CONTEXT_LINES)
            end = min(len(lines), index + DIAGNOSTIC_CONTEXT_LINES + 1)
            selected_indexes.update(range(start, end))

    if not selected_indexes:
        selected_indexes.update(
            range(max(0, len(lines) - (DIAGNOSTIC_CONTEXT_LINES * 2 + 1)), len(lines))
        )

    selected_lines = []
    previous_index = None
    for index in sorted(selected_indexes):
        if previous_index is not None and index > previous_index + 1:
            selected_lines.append("...")
        selected_lines.append(lines[index])
        previous_index = index
    return _bounded_redacted_text("\n".join(selected_lines), max_chars=max_chars)


def _contains_log_file(files: list[Path]) -> bool:
    log_suffixes = {".err", ".log", ".out", ".txt"}
    return any(
        path.suffix.lower() in log_suffixes
        or "stderr" in path.name.lower()
        or "stdout" in path.name.lower()
        for path in files
    )


def _bounded_redacted_text(
    value: str,
    max_chars: int = MAX_DIAGNOSTIC_LINE_CHARS,
) -> str:
    redacted = SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1\2[REDACTED]", value)
    redacted = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = JWT_PATTERN.sub("[REDACTED JWT]", redacted)
    redacted = SAS_PARAMETER_PATTERN.sub(r"\1[REDACTED]", redacted)
    if len(redacted) <= max_chars:
        return redacted
    return f"{redacted[:max_chars]}\n... [diagnostic output truncated]"


def _safe_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "unknown-job"


def get_registered_model(
    ml_client: MLClient,
    model_name: str,
    model_version: str,
    require_mlflow: bool = False,
) -> Any:
    try:
        model = ml_client.models.get(name=model_name, version=model_version)
    except Exception as exc:
        raise RuntimeError(
            "Registered model "
            f"'{model_name}:{model_version}' was not found. Run the training/model "
            "registration workflow first or provide an existing model name and version."
        ) from exc
    if require_mlflow and str(model.type).lower() != "mlflow_model":
        raise RuntimeError(
            f"Registered model '{model_name}:{model_version}' has type "
            f"'{model.type}'. The reusable no-code deployment workflows require an "
            "MLflow model; register it with model_type=mlflow_model."
        )
    return model


def get_kubernetes_online_compute(ml_client: MLClient, compute_name: str) -> Any:
    try:
        compute = ml_client.compute.get(compute_name)
    except Exception as exc:
        raise RuntimeError(
            f"Attached Azure ML compute '{compute_name}' was not found. "
            "Provision a supported Kubernetes cluster, install the Azure ML "
            "extension, and attach it to the workspace. For direct private AKS "
            "with local accounts disabled, infrastructure must first create the "
            "per-workspace Trusted Access mlworkload role binding; Azure "
            "Arc-enabled Kubernetes is the fallback."
        ) from exc

    compute_type = str(getattr(compute, "type", "")).lower()
    if compute_type != "kubernetes":
        raise RuntimeError(
            f"Azure ML compute '{compute_name}' has type "
            f"'{getattr(compute, 'type', None)}'; the private online workflow "
            "requires an attached Kubernetes compute."
        )

    resource_id = str(getattr(compute, "resource_id", "") or "")
    supported_cluster_resource_types = (
        "/providers/microsoft.containerservice/managedclusters/",
        "/providers/microsoft.kubernetes/connectedclusters/",
    )
    if not any(
        resource_type in resource_id.lower()
        for resource_type in supported_cluster_resource_types
    ):
        raise RuntimeError(
            f"Azure ML Kubernetes compute '{compute_name}' must be backed by a "
            "direct AKS managedClusters resource or an Azure Arc-enabled "
            "Kubernetes connectedClusters resource."
        )

    provisioning_state = str(getattr(compute, "provisioning_state", ""))
    if provisioning_state not in SUCCESS_PROVISIONING_STATES:
        raise RuntimeError(
            f"Azure ML Kubernetes compute '{compute_name}' is not ready; "
            f"provisioning state is '{provisioning_state}'."
        )

    namespace = str(getattr(compute, "namespace", "")).strip()
    if not namespace or namespace == "default":
        raise RuntimeError(
            f"Azure ML Kubernetes compute '{compute_name}' must use a dedicated "
            "non-default namespace configured by infrastructure."
        )

    identity = getattr(compute, "identity", None)
    identity_type = str(getattr(identity, "type", "")).lower().replace("_", "")
    user_assigned_identities = getattr(identity, "user_assigned_identities", None)
    if not identity_type.endswith("userassigned") or not user_assigned_identities:
        raise RuntimeError(
            f"Azure ML Kubernetes compute '{compute_name}' must use an "
            "infrastructure-supplied user-assigned managed identity. AKS node "
            "identity fallback is not supported."
        )
    return compute


def get_prebuilt_environment(
    ml_client: MLClient,
    environment_name: str,
    environment_version: str,
) -> Any:
    try:
        environment = ml_client.environments.get(
            name=environment_name,
            version=environment_version,
        )
    except Exception as exc:
        raise RuntimeError(
            "Registered inference environment "
            f"'{environment_name}:{environment_version}' was not found. "
            "Register the immutable prebuilt environment before deployment."
        ) from exc

    image = str(getattr(environment, "image", "") or "")
    if not IMAGE_DIGEST_PATTERN.search(image):
        raise RuntimeError(
            "Registered inference environment "
            f"'{environment_name}:{environment_version}' must reference a "
            "prebuilt container image by sha256 digest."
        )
    if getattr(environment, "build", None) is not None or getattr(
        environment, "conda_file", None
    ):
        raise RuntimeError(
            "Registered inference environment "
            f"'{environment_name}:{environment_version}' includes a build "
            "context or Conda specification. Private online deployment requires "
            "an image-only environment so Azure ML never starts an image build."
        )
    return environment
