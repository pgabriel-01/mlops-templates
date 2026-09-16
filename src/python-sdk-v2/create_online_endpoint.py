# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import KubernetesOnlineEndpoint

from aml_client import (
    add_workspace_arguments,
    create_ml_client,
    get_kubernetes_online_compute,
    get_user_assigned_identity_configuration,
    wait_for_resource_create_or_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update a Kubernetes online endpoint."
    )
    add_workspace_arguments(parser)
    parser.add_argument("--endpoint_name", required=True)
    parser.add_argument("--compute", required=True)
    parser.add_argument("--description")
    parser.add_argument("--auth_mode", default="aml_token")
    parser.add_argument("--endpoint_uami_resource_id")
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    compute = get_kubernetes_online_compute(ml_client, args.compute)
    endpoint_arguments = {
        "name": args.endpoint_name,
        "description": args.description,
        "auth_mode": args.auth_mode,
        "compute": compute.id,
    }
    identity = get_user_assigned_identity_configuration(
        args.endpoint_uami_resource_id
    )
    if identity is not None:
        endpoint_arguments["identity"] = identity
    endpoint = KubernetesOnlineEndpoint(
        **endpoint_arguments,
    )
    return wait_for_resource_create_or_update(
        lambda: ml_client.online_endpoints.begin_create_or_update(endpoint),
        lambda: ml_client.online_endpoints.get(args.endpoint_name),
        f"Kubernetes online endpoint '{args.endpoint_name}'",
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
