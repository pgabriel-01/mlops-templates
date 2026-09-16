# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import Data

from aml_client import add_workspace_arguments, create_ml_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a data asset.")
    add_workspace_arguments(parser)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--description")
    parser.add_argument("--data_type", default="uri_file")
    parser.add_argument("--data_path", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    data = Data(
        path=args.data_path,
        type=args.data_type,
        description=args.description,
        name=args.data_name,
    )
    return ml_client.data.create_or_update(data)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
