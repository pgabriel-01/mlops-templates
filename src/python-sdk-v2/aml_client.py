# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential, DefaultAzureCredential

AZUREML_CREDENTIAL_MODE = "AZUREML_CREDENTIAL_MODE"
AZURE_CLI_CREDENTIAL_MODE = "azure-cli"
DEFAULT_CREDENTIAL_MODE = "default"
MANAGEMENT_SCOPE = "https://management.azure.com/.default"
SUCCESS_JOB_STATUSES = {"Completed"}
FAILED_JOB_STATUSES = {"Failed", "Canceled", "Cancelled", "NotResponding"}
TERMINAL_JOB_STATUSES = {
    "Completed",
    "Failed",
    "Canceled",
    "Cancelled",
    "NotResponding",
    "Paused",
}
DIAGNOSTICS_DIRECTORY = "aml-diagnostics"
MAX_DIAGNOSTIC_FILES = 20
MAX_DIAGNOSTIC_FILE_BYTES = 256_000
MAX_DIAGNOSTIC_OUTPUT_CHARS = 20_000
MAX_DIAGNOSTIC_LINE_CHARS = 2_000
DIAGNOSTIC_CONTEXT_LINES = 4
ERROR_LINE_PATTERN = re.compile(
    r"traceback|exception|error|failed|fatal|stderr|exit code",
    re.IGNORECASE,
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b("
    r"access[_-]?token|api[_-]?key|authorization|client[_-]?secret|credential|"
    r"password|private[_-]?key|secret|token"
    r")(\s*[:=]\s*)([^\r\n,;]+)"
)
BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
SAS_PARAMETER_PATTERN = re.compile(
    r"(?i)([?&](?:sig|se|sp|spr|sr|sv|st)=)[^&\s]+"
)


def add_workspace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)


def create_azure_credential() -> Any:
    is_ci = _is_truthy(os.environ.get("GITHUB_ACTIONS")) or _is_truthy(
        os.environ.get("CI")
    )
    mode = os.environ.get(AZUREML_CREDENTIAL_MODE)
    if mode is None:
        mode = AZURE_CLI_CREDENTIAL_MODE if is_ci else DEFAULT_CREDENTIAL_MODE
    mode = mode.strip().lower()

    if mode not in {AZURE_CLI_CREDENTIAL_MODE, DEFAULT_CREDENTIAL_MODE}:
        raise RuntimeError(
            f"Unsupported {AZUREML_CREDENTIAL_MODE} value '{mode}'. "
            f"Use '{AZURE_CLI_CREDENTIAL_MODE}' or '{DEFAULT_CREDENTIAL_MODE}'."
        )
    if is_ci and mode != AZURE_CLI_CREDENTIAL_MODE:
        raise RuntimeError(
            f"{AZUREML_CREDENTIAL_MODE} must be '{AZURE_CLI_CREDENTIAL_MODE}' in CI. "
            "Run azure/login before the Python SDK step; managed identity fallback "
            "is intentionally disabled."
        )

    if mode == AZURE_CLI_CREDENTIAL_MODE:
        credential = AzureCliCredential()
        credential_description = "Azure CLI credential created by azure/login"
    else:
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=True,
            exclude_managed_identity_credential=True,
        )
        credential_description = "local DefaultAzureCredential"

    try:
        credential.get_token(MANAGEMENT_SCOPE)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to acquire an Azure management token with "
            f"{credential_description}. "
            + (
                "Verify that azure/login completed successfully in this job."
                if mode == AZURE_CLI_CREDENTIAL_MODE
                else "Sign in with Azure CLI or configure a supported local credential."
            )
        ) from exc
    return credential


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes"}


def create_ml_client(args: argparse.Namespace) -> MLClient:
    credential = create_azure_credential()
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
        print(f"Job error: {_bounded_redacted_text(str(error))}", flush=True)

    try:
        children = list(ml_client.jobs.list(parent_job_name=job.name))
    except Exception as exc:
        print(
            "Unable to list child-job diagnostics: "
            f"{_bounded_redacted_text(str(exc))}",
            flush=True,
        )
        children = []

    failed_children = []
    for child in children:
        child_error = getattr(child, "error", None)
        print(
            f"Child job {child.name}: status={child.status}"
            + (
                f", error={_bounded_redacted_text(str(child_error))}"
                if child_error
                else ""
            ),
            flush=True,
        )
        if str(child.status) in FAILED_JOB_STATUSES:
            failed_children.append(child)

    diagnostic_jobs = failed_children or [job]
    diagnostics_root = Path(
        os.environ.get("AML_DIAGNOSTICS_DIR", DIAGNOSTICS_DIRECTORY)
    )
    parent_directory = diagnostics_root / _safe_path_component(str(job.name))
    print(f"AML diagnostics directory: {parent_directory}", flush=True)
    for diagnostic_job in diagnostic_jobs:
        _download_and_print_job_diagnostics(
            ml_client,
            diagnostic_job,
            parent_directory,
        )


def _download_and_print_job_diagnostics(
    ml_client: MLClient,
    job: Any,
    parent_directory: Path,
) -> None:
    job_name = str(job.name)
    download_directory = parent_directory / _safe_path_component(job_name)
    download_directory.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading diagnostics for job {job_name} to {download_directory}",
        flush=True,
    )
    try:
        ml_client.jobs.download(
            name=job_name,
            download_path=str(download_directory),
            all=False,
        )
    except Exception as exc:
        print(
            f"Unable to download diagnostics for job {job_name}: "
            f"{_bounded_redacted_text(str(exc))}",
            flush=True,
        )
        return

    files = sorted(path for path in download_directory.rglob("*") if path.is_file())
    if not _contains_log_file(files):
        print(
            f"No log files found in standard diagnostics for job {job_name}; "
            "retrying full job download",
            flush=True,
        )
        try:
            ml_client.jobs.download(
                name=job_name,
                download_path=str(download_directory),
                all=True,
            )
        except Exception as exc:
            print(
                f"Unable to download full diagnostics for job {job_name}: "
                f"{_bounded_redacted_text(str(exc))}",
                flush=True,
            )
        files = sorted(
            path for path in download_directory.rglob("*") if path.is_file()
        )

    if not files:
        print(f"No diagnostic files downloaded for job {job_name}", flush=True)
        return

    remaining_chars = MAX_DIAGNOSTIC_OUTPUT_CHARS
    for path in files[:MAX_DIAGNOSTIC_FILES]:
        if remaining_chars <= 0:
            break
        try:
            with path.open("rb") as diagnostic_file:
                content = diagnostic_file.read(MAX_DIAGNOSTIC_FILE_BYTES).decode(
                    "utf-8",
                    errors="replace",
                )
        except Exception as exc:
            print(
                f"Unable to read diagnostic file {path}: "
                f"{_bounded_redacted_text(str(exc))}",
                flush=True,
            )
            continue

        excerpt = _diagnostic_excerpt(content, remaining_chars)
        if not excerpt:
            continue
        relative_path = path.relative_to(download_directory)
        print(f"--- Diagnostics from {job_name}/{relative_path} ---", flush=True)
        print(excerpt, flush=True)
        remaining_chars -= len(excerpt)

    if len(files) > MAX_DIAGNOSTIC_FILES:
        print(
            f"Diagnostic file output limited to {MAX_DIAGNOSTIC_FILES} of "
            f"{len(files)} files for job {job_name}",
            flush=True,
        )
    if remaining_chars <= 0:
        print(
            f"Diagnostic output limited to {MAX_DIAGNOSTIC_OUTPUT_CHARS} "
            f"characters for job {job_name}",
            flush=True,
        )


def _diagnostic_excerpt(content: str, max_chars: int) -> str:
    lines = content.splitlines()
    selected_indexes = set()
    for index, line in enumerate(lines):
        if ERROR_LINE_PATTERN.search(line):
            start = max(0, index - DIAGNOSTIC_CONTEXT_LINES)
            end = min(len(lines), index + DIAGNOSTIC_CONTEXT_LINES + 1)
            selected_indexes.update(range(start, end))

    if not selected_indexes:
        selected_indexes.update(
            range(max(0, len(lines) - (DIAGNOSTIC_CONTEXT_LINES * 2 + 1)), len(lines))
        )

    selected_lines = []
    previous_index = None
    for index in sorted(selected_indexes):
        if previous_index is not None and index > previous_index + 1:
            selected_lines.append("...")
        selected_lines.append(lines[index])
        previous_index = index
    return _bounded_redacted_text("\n".join(selected_lines), max_chars=max_chars)


def _contains_log_file(files: list[Path]) -> bool:
    log_suffixes = {".err", ".log", ".out", ".txt"}
    return any(
        path.suffix.lower() in log_suffixes
        or "stderr" in path.name.lower()
        or "stdout" in path.name.lower()
        for path in files
    )


def _bounded_redacted_text(
    value: str,
    max_chars: int = MAX_DIAGNOSTIC_LINE_CHARS,
) -> str:
    redacted = SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1\2[REDACTED]", value)
    redacted = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = JWT_PATTERN.sub("[REDACTED JWT]", redacted)
    redacted = SAS_PARAMETER_PATTERN.sub(r"\1[REDACTED]", redacted)
    if len(redacted) <= max_chars:
        return redacted
    return f"{redacted[:max_chars]}\n... [diagnostic output truncated]"


def _safe_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "unknown-job"


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
