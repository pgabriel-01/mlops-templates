# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os

from azure.ai.ml import load_job
from azure.ai.ml.entities import Model

from aml_client import (
    add_workspace_arguments,
    create_ml_client,
    wait_for_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Azure ML training job and register its model output."
    )
    add_workspace_arguments(parser)
    parser.add_argument("--job_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_output_name", default="model")
    parser.add_argument("--model_type", default="custom_model")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Model:
    ml_client = create_ml_client(args)
    submitted_job = ml_client.jobs.create_or_update(load_job(source=args.job_file))
    completed_job = wait_for_job(ml_client, submitted_job.name)
    model_path = (
        f"azureml://jobs/{completed_job.name}/outputs/"
        f"{args.model_output_name}/paths/"
    )
    model = Model(
        name=args.model_name,
        path=model_path,
        type=args.model_type,
    )
    registered_model = ml_client.models.create_or_update(model)
    print(
        f"Registered model {registered_model.name}:{registered_model.version}",
        flush=True,
    )
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"training_job_name={completed_job.name}\n")
            output.write(f"model_name={registered_model.name}\n")
            output.write(f"model_version={registered_model.version}\n")
    return registered_model


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
