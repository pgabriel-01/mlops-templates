import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SDK_PATH = Path(__file__).parents[1] / "src" / "python-sdk-v2"
sys.path.insert(0, str(SDK_PATH))

aml_client = importlib.import_module("aml_client")
create_online_deployment = importlib.import_module("create_online_deployment")
test_batch_endpoint = importlib.import_module("test_batch_endpoint")
train_and_register_model = importlib.import_module("train_and_register_model")


def test_create_ml_client_uses_explicit_coordinates_and_noninteractive_credential(
    monkeypatch,
):
    credential = Mock()
    credential_type = Mock(return_value=credential)
    client_type = Mock(return_value=Mock())
    monkeypatch.setattr(aml_client, "DefaultAzureCredential", credential_type)
    monkeypatch.setattr(aml_client, "MLClient", client_type)
    args = argparse.Namespace(
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
    )

    aml_client.create_ml_client(args)

    credential_type.assert_called_once_with(
        exclude_interactive_browser_credential=True
    )
    credential.get_token.assert_called_once_with(
        "https://management.azure.com/.default"
    )
    client_type.assert_called_once_with(
        credential=credential,
        subscription_id="sub",
        resource_group_name="rg",
        workspace_name="ws",
    )


def test_online_traffic_update_uses_online_endpoint_operation(monkeypatch):
    client = Mock()
    client.models.get.return_value = SimpleNamespace(id="azureml:model:1")
    client.online_deployments.begin_create_or_update.return_value.result.return_value = (
        Mock()
    )
    endpoint = SimpleNamespace(traffic={})
    client.online_endpoints.get.return_value = endpoint
    client.online_endpoints.begin_create_or_update.return_value.result.return_value = (
        endpoint
    )
    monkeypatch.setattr(create_online_deployment, "create_ml_client", lambda _: client)
    args = argparse.Namespace(
        deployment_name="blue",
        endpoint_name="endpoint",
        model_name="model",
        model_version="1",
        instance_type="Standard_DS2_v2",
        instance_count=1,
        traffic_allocation=100,
    )

    create_online_deployment.run(args)

    assert endpoint.traffic == {"blue": 100}
    client.online_endpoints.begin_create_or_update.assert_called_once_with(endpoint)


def test_missing_registered_model_has_actionable_error():
    client = Mock()
    client.models.get.side_effect = RuntimeError("not found")

    with pytest.raises(RuntimeError, match="training/model registration workflow"):
        aml_client.get_registered_model(client, "model", "7")


def test_batch_invoke_waits_for_terminal_success(monkeypatch):
    client = Mock()
    client.batch_endpoints.invoke.return_value = SimpleNamespace(name="batch-job")
    waiter = Mock(return_value=SimpleNamespace(status="Completed"))
    monkeypatch.setattr(test_batch_endpoint, "create_ml_client", lambda _: client)
    monkeypatch.setattr(test_batch_endpoint, "wait_for_job", waiter)
    args = argparse.Namespace(
        endpoint_name="batch-endpoint",
        request_batch_file="azureml://datastores/test/paths/input",
        request_type="uri_folder",
    )

    test_batch_endpoint.run(args)

    waiter.assert_called_once_with(client, "batch-job")


def test_failed_job_surfaces_parent_and_child_diagnostics(monkeypatch, capsys):
    failed = SimpleNamespace(name="parent", status="Failed", error="parent error")
    child = SimpleNamespace(name="child", status="Failed", error="child error")
    client = Mock()
    client.jobs.get.return_value = failed
    client.jobs.list.return_value = [child]
    monkeypatch.setattr(aml_client.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="finished with status Failed"):
        aml_client.wait_for_job(client, "parent", poll_interval_seconds=0)

    output = capsys.readouterr().out
    assert "parent error" in output
    assert "Child job child: status=Failed, error=child error" in output


def test_training_waits_then_registers_job_output(monkeypatch):
    client = Mock()
    client.jobs.create_or_update.return_value = SimpleNamespace(name="training-job")
    client.models.create_or_update.return_value = SimpleNamespace(
        name="forecast",
        version="3",
    )
    monkeypatch.setattr(train_and_register_model, "create_ml_client", lambda _: client)
    monkeypatch.setattr(
        train_and_register_model,
        "load_job",
        Mock(return_value="loaded-job"),
    )
    monkeypatch.setattr(
        train_and_register_model,
        "wait_for_job",
        Mock(return_value=SimpleNamespace(name="training-job")),
    )
    args = argparse.Namespace(
        job_file="jobs/train.yml",
        model_name="forecast",
        model_output_name="model",
        model_type="custom_model",
    )

    train_and_register_model.run(args)

    registered = client.models.create_or_update.call_args.args[0]
    assert registered.name == "forecast"
    assert registered.path == "azureml://jobs/training-job/outputs/model/paths/"


def test_training_writes_reusable_workflow_outputs(monkeypatch, tmp_path):
    client = Mock()
    client.jobs.create_or_update.return_value = SimpleNamespace(name="training-job")
    client.models.create_or_update.return_value = SimpleNamespace(
        name="forecast",
        version="3",
    )
    output_file = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(train_and_register_model, "create_ml_client", lambda _: client)
    monkeypatch.setattr(train_and_register_model, "load_job", lambda source: source)
    monkeypatch.setattr(
        train_and_register_model,
        "wait_for_job",
        lambda *_: SimpleNamespace(name="training-job"),
    )
    args = argparse.Namespace(
        job_file="jobs/train.yml",
        model_name="forecast",
        model_output_name="model",
        model_type="custom_model",
    )

    train_and_register_model.run(args)

    assert output_file.read_text(encoding="utf-8").splitlines() == [
        "training_job_name=training-job",
        "model_name=forecast",
        "model_version=3",
    ]
