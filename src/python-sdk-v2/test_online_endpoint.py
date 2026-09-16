# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from aml_client import (
    add_workspace_arguments,
    create_ml_client,
    use_private_ca_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke an online endpoint.")
    add_workspace_arguments(parser)
    parser.add_argument("--endpoint_name", required=True)
    parser.add_argument("--request_file", required=True)
    parser.add_argument("--ca_bundle", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace):
    with use_private_ca_bundle(args.ca_bundle):
        ml_client = create_ml_client(args)
        response = ml_client.online_endpoints.invoke(
            endpoint_name=args.endpoint_name,
            request_file=args.request_file,
        )
    print(response, flush=True)
    return response


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
