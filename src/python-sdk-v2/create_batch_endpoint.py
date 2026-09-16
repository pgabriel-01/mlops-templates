# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import BatchEndpoint

from aml_client import add_workspace_arguments, create_ml_client, wait_for_poller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a batch endpoint.")
    add_workspace_arguments(parser)
    parser.add_argument("--endpoint_name", required=True)
    parser.add_argument("--description")
    parser.add_argument("--auth_mode", default="aad_token")
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    endpoint = BatchEndpoint(
        name=args.endpoint_name,
        description=args.description,
        auth_mode=args.auth_mode,
    )
    return wait_for_poller(
        ml_client.batch_endpoints.begin_create_or_update(endpoint)
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
