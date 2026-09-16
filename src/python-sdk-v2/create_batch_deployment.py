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
    get_registered_model,
    wait_for_resource_create_or_update,
)

DEFAULT_BATCH_ENVIRONMENT = (
    "azureml://registries/azureml/environments/sklearn-1.5/versions/53"
)
DEFAULT_BATCH_CODE_PATH = Path(__file__).with_name("batch_scoring")
DEFAULT_BATCH_SCORING_SCRIPT = "score.py"
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


def create_code_configuration(
    code_path: str | Path,
    scoring_script: str,
) -> CodeConfiguration:
    code_directory = Path(code_path).expanduser().resolve()
    normalized_scoring_script = scoring_script.strip()
    if not normalized_scoring_script:
        raise argparse.ArgumentTypeError("A batch scoring script is required.")

    scoring_path = (code_directory / normalized_scoring_script).resolve()
    try:
        scoring_path.relative_to(code_directory)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Batch scoring script must be inside the configured code directory."
        ) from exc

    if not code_directory.is_dir():
        raise argparse.ArgumentTypeError(
            f"Batch scoring code directory does not exist: {code_directory}"
        )
    if not scoring_path.is_file():
        raise argparse.ArgumentTypeError(
            f"Batch scoring script does not exist: {scoring_path}"
        )

    return CodeConfiguration(
        code=str(code_directory),
        scoring_script=scoring_path.relative_to(code_directory).as_posix(),
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
        "--code_path",
        default=str(DEFAULT_BATCH_CODE_PATH),
        help="Directory containing the batch scoring source.",
    )
    parser.add_argument(
        "--scoring_script",
        default=DEFAULT_BATCH_SCORING_SCRIPT,
        help="Scoring script path relative to --code_path.",
    )
    parser.add_argument("--instance_count", type=int, default=2)
    parser.add_argument("--max_concurrency_per_instance", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--output_file_name", default="predictions.csv")
    return parser.parse_args()


def run(args: argparse.Namespace):
    environment = validate_immutable_environment_reference(args.environment)
    code_configuration = create_code_configuration(
        args.code_path,
        args.scoring_script,
    )
    ml_client = create_ml_client(args)
    model = get_registered_model(
        ml_client,
        args.model_name,
        args.model_version,
        require_mlflow=True,
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
    wait_for_resource_create_or_update(
        lambda: ml_client.batch_deployments.begin_create_or_update(deployment),
        lambda: ml_client.batch_deployments.get(
            args.deployment_name,
            endpoint_name=args.endpoint_name,
        ),
        f"batch deployment {args.deployment_name}",
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
