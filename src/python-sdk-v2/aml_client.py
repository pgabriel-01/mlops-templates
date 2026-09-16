# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import time
from typing import Any

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

SUCCESS_JOB_STATUSES = {"Completed"}
TERMINAL_JOB_STATUSES = {
    "Completed",
    "Failed",
    "Canceled",
    "Cancelled",
    "NotResponding",
    "Paused",
}


def add_workspace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)


def create_ml_client(args: argparse.Namespace) -> MLClient:
    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=True,
    )
    credential.get_token("https://management.azure.com/.default")
    return MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )


def wait_for_poller(poller: Any) -> Any:
    return poller.result()


def wait_for_job(
    ml_client: MLClient,
    job_name: str,
    poll_interval_seconds: int = 15,
) -> Any:
    while True:
        job = ml_client.jobs.get(job_name)
        status = str(job.status)
        print(f"Job {job_name} status: {status}", flush=True)
        if status in TERMINAL_JOB_STATUSES:
            break
        time.sleep(poll_interval_seconds)

    if status not in SUCCESS_JOB_STATUSES:
        print_job_diagnostics(ml_client, job)
        raise RuntimeError(f"Azure ML job {job_name} finished with status {status}")
    return job


def print_job_diagnostics(ml_client: MLClient, job: Any) -> None:
    error = getattr(job, "error", None)
    if error:
        print(f"Job error: {error}", flush=True)

    try:
        children = list(ml_client.jobs.list(parent_job_name=job.name))
    except Exception as exc:
        print(f"Unable to list child-job diagnostics: {exc}", flush=True)
        children = []

    for child in children:
        child_error = getattr(child, "error", None)
        print(
            f"Child job {child.name}: status={child.status}"
            + (f", error={child_error}" if child_error else ""),
            flush=True,
        )


def get_registered_model(
    ml_client: MLClient,
    model_name: str,
    model_version: str,
    require_mlflow: bool = False,
) -> Any:
    try:
        model = ml_client.models.get(name=model_name, version=model_version)
    except Exception as exc:
        raise RuntimeError(
            "Registered model "
            f"'{model_name}:{model_version}' was not found. Run the training/model "
            "registration workflow first or provide an existing model name and version."
        ) from exc
    if require_mlflow and str(model.type).lower() != "mlflow_model":
        raise RuntimeError(
            f"Registered model '{model_name}:{model_version}' has type "
            f"'{model.type}'. The reusable no-code deployment workflows require an "
            "MLflow model; register it with model_type=mlflow_model."
        )
    return model
