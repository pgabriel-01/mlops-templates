# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import KubernetesOnlineDeployment

from aml_client import (
    add_workspace_arguments,
    create_ml_client,
    get_prebuilt_environment,
    get_registered_model,
    wait_for_resource_create_or_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update an online deployment and its traffic."
    )
    add_workspace_arguments(parser)
    parser.add_argument("--deployment_name", required=True)
    parser.add_argument("--endpoint_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--environment_name", required=True)
    parser.add_argument("--environment_version", required=True)
    parser.add_argument("--instance_type", required=True)
    parser.add_argument("--instance_count", type=int, default=1)
    parser.add_argument("--traffic_allocation", type=int, default=100)
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    model = get_registered_model(
        ml_client,
        args.model_name,
        args.model_version,
        require_mlflow=True,
    )
    environment = get_prebuilt_environment(
        ml_client,
        args.environment_name,
        args.environment_version,
    )
    deployment = KubernetesOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=model.id,
        environment=environment.id,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )
    wait_for_resource_create_or_update(
        lambda: ml_client.online_deployments.begin_create_or_update(deployment),
        lambda: ml_client.online_deployments.get(
            name=args.deployment_name,
            endpoint_name=args.endpoint_name,
        ),
        f"Kubernetes online deployment '{args.deployment_name}'",
    )

    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {args.deployment_name: args.traffic_allocation}
    return wait_for_resource_create_or_update(
        lambda: ml_client.online_endpoints.begin_create_or_update(endpoint),
        lambda: ml_client.online_endpoints.get(args.endpoint_name),
        f"Kubernetes online endpoint '{args.endpoint_name}' traffic",
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
