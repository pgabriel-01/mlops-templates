# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import re
from pathlib import Path

from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.ai.ml.entities import BatchDeployment, CodeConfiguration

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
    re.compile(
        r"azureml://registries/[^/\s]+/environments/[^/\s]+/versions/\d+"
    ),
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
FULL_ENVIRONMENT_ID_PATTERN = re.compile(
    r"/subscriptions/[^/\s]+/resourceGroups/[^/\s]+/providers/"
    r"Microsoft\.MachineLearningServices/(?:workspaces|registries)/[^/\s]+/"
    r"environments/[^/\s]+/versions/\d+",
    re.IGNORECASE,
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


def resolve_batch_environment(ml_client: object, reference: str) -> str:
    registry_match = REGISTRY_ENVIRONMENT_PATTERN.fullmatch(reference)
    if not registry_match:
        return reference

    registry_name, environment_name, version = registry_match.groups()
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

    resolved_id = str(getattr(environment, "id", "") or "")
    if not FULL_ENVIRONMENT_ID_PATTERN.fullmatch(resolved_id) or not (
        _environment_matches(reference, resolved_id)
    ):
        raise RuntimeError(
            "Azure ML returned an invalid or mismatched resource ID for immutable "
            f"registry environment '{reference}'. resolved={resolved_id!r}"
        )
    return resolved_id


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
        raise ValueError(
            f"Scoring code directory does not exist: {code_directory}"
        )

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
    requested_environment: str,
    scoring_script: str,
) -> None:
    live_environment = getattr(deployment, "environment", None)
    if not live_environment or not _environment_matches(
        requested_environment,
        str(live_environment),
    ):
        raise RuntimeError(
            "Azure ML batch deployment did not persist the requested immutable "
            f"environment. requested={requested_environment!r}, "
            f"live={live_environment!r}"
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
