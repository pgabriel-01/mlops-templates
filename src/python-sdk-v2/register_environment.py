# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import BuildContext, Environment

from aml_client import add_workspace_arguments, create_ml_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register an environment.")
    add_workspace_arguments(parser)
    parser.add_argument("--environment_name", required=True)
    parser.add_argument("--description")
    parser.add_argument("--env_path", required=True)
    parser.add_argument("--build_type", choices=("docker", "conda"), required=True)
    parser.add_argument(
        "--base_image",
        default="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    )
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    if args.build_type == "docker":
        environment = Environment(
            name=args.environment_name,
            build=BuildContext(path=args.env_path),
            description=args.description,
        )
    else:
        environment = Environment(
            image=args.base_image,
            conda_file=args.env_path,
            name=args.environment_name,
            description=args.description,
        )
    return ml_client.environments.create_or_update(environment)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
