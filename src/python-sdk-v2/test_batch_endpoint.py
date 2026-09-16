# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml import Input
from azure.ai.ml.constants import InputOutputModes

from aml_client import add_workspace_arguments, create_ml_client, wait_for_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke a batch endpoint and wait for completion."
    )
    add_workspace_arguments(parser)
    parser.add_argument("--endpoint_name", required=True)
    parser.add_argument("--request_batch_file", required=True)
    parser.add_argument(
        "--request_type",
        choices=("uri_folder", "uri_file"),
        required=True,
    )
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    invocation = ml_client.batch_endpoints.invoke(
        endpoint_name=args.endpoint_name,
        input=Input(
            path=args.request_batch_file,
            type=args.request_type,
            mode=InputOutputModes.DOWNLOAD,
        ),
    )
    return wait_for_job(ml_client, invocation.name)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
