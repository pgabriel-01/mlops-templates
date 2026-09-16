# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.ai.ml.entities import BatchDeployment

from aml_client import (
    add_workspace_arguments,
    create_ml_client,
    get_registered_model,
    wait_for_poller,
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
    parser.add_argument("--instance_count", type=int, default=2)
    parser.add_argument("--max_concurrency_per_instance", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--output_file_name", default="predictions.csv")
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    model = get_registered_model(
        ml_client,
        args.model_name,
        args.model_version,
    )
    deployment = BatchDeployment(
        name=args.deployment_name,
        description=args.description,
        endpoint_name=args.endpoint_name,
        model=model.id,
        compute=args.compute,
        instance_count=args.instance_count,
        max_concurrency_per_instance=args.max_concurrency_per_instance,
        mini_batch_size=args.mini_batch_size,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name=args.output_file_name,
    )
    wait_for_poller(
        ml_client.batch_deployments.begin_create_or_update(deployment)
    )

    endpoint = ml_client.batch_endpoints.get(args.endpoint_name)
    endpoint.defaults.deployment_name = args.deployment_name
    return wait_for_poller(
        ml_client.batch_endpoints.begin_create_or_update(endpoint)
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
