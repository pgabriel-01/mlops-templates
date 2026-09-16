import argparse
import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import yaml
from azure.core.exceptions import ResourceExistsError

SDK_PATH = Path(__file__).parents[1] / "src" / "python-sdk-v2"
sys.path.insert(0, str(SDK_PATH))

aml_client = importlib.import_module("aml_client")
create_batch_deployment = importlib.import_module("create_batch_deployment")
create_batch_endpoint = importlib.import_module("create_batch_endpoint")
create_online_deployment = importlib.import_module("create_online_deployment")
create_online_endpoint = importlib.import_module("create_online_endpoint")
test_batch_endpoint = importlib.import_module("test_batch_endpoint")
train_and_register_model = importlib.import_module("train_and_register_model")

@pytest.mark.parametrize("ci_environment", ["GITHUB_ACTIONS", "CI"])
def test_ci_uses_validated_azure_cli_credential(monkeypatch, ci_environment):
    credential = Mock()
    cli_credential_type = Mock(return_value=credential)
    default_credential_type = Mock()
    client_type = Mock(return_value=Mock())
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv(ci_environment, "true")
    monkeypatch.delenv("AZUREML_CREDENTIAL_MODE", raising=False)
    monkeypatch.setattr(aml_client, "AzureCliCredential", cli_credential_type)
    monkeypatch.setattr(aml_client, "DefaultAzureCredential", default_credential_type)
    monkeypatch.setattr(aml_client, "MLClient", client_type)
    args = argparse.Namespace(
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
    )
    aml_client.create_ml_client(args)

    cli_credential_type.assert_called_once_with()
    default_credential_type.assert_not_called()
    credential.get_token.assert_called_once_with(
        "https://management.azure.com/.default"
    )
    client_type.assert_called_once_with(
        credential=credential,
        subscription_id="sub",
        resource_group_name="rg",
        workspace_name="ws",
    )


def test_local_mode_uses_default_credential_without_managed_identity(monkeypatch):
    credential = Mock()
    cli_credential_type = Mock()
    default_credential_type = Mock(return_value=credential)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("AZUREML_CREDENTIAL_MODE", raising=False)
    monkeypatch.setattr(aml_client, "AzureCliCredential", cli_credential_type)
    monkeypatch.setattr(aml_client, "DefaultAzureCredential", default_credential_type)

    result = aml_client.create_azure_credential()

    assert result is credential
    cli_credential_type.assert_not_called()
    default_credential_type.assert_called_once_with(
        exclude_interactive_browser_credential=True,
        exclude_managed_identity_credential=True,
    )
    credential.get_token.assert_called_once_with(
        "https://management.azure.com/.default"
    )


def test_ci_rejects_default_mode_before_constructing_any_credential(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("AZUREML_CREDENTIAL_MODE", "default")
    cli_credential_type = Mock()
    default_credential_type = Mock()
    monkeypatch.setattr(aml_client, "AzureCliCredential", cli_credential_type)
    monkeypatch.setattr(aml_client, "DefaultAzureCredential", default_credential_type)

    with pytest.raises(RuntimeError, match="managed identity fallback"):
        aml_client.create_azure_credential()

    cli_credential_type.assert_not_called()
    default_credential_type.assert_not_called()


def test_credential_token_validation_failure_is_explicit(monkeypatch):
    credential = Mock()
    credential.get_token.side_effect = RuntimeError("not logged in")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("AZUREML_CREDENTIAL_MODE", raising=False)
    monkeypatch.setattr(
        aml_client,
        "AzureCliCredential",
        Mock(return_value=credential),
    )

    with pytest.raises(RuntimeError, match="azure/login completed successfully"):
        aml_client.create_azure_credential()

    credential.get_token.assert_called_once_with(
        "https://management.azure.com/.default"
    )


def test_reusable_workflows_force_azure_cli_credential_mode():
    repository_root = Path(__file__).parents[1]

    for workflow_name in (
        "python-sdk-v2-train-register.yml",
        "python-sdk-v2-online.yml",
        "python-sdk-v2-batch.yml",
    ):
        workflow = (
            repository_root / ".github" / "workflows" / workflow_name
        ).read_text(encoding="utf-8")
        assert "AZUREML_CREDENTIAL_MODE: azure-cli" in workflow


def test_online_workflow_requires_private_kubernetes_contract():
    workflow_path = (
        Path(__file__).parents[1]
        / ".github"
        / "workflows"
        / "python-sdk-v2-online.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    inputs = parsed[True]["workflow_call"]["inputs"]

    for required_input in (
        "runner",
        "compute",
        "environment_name",
        "environment_version",
        "instance_type",
        "instance_count",
    ):
        assert required_input in inputs
    assert inputs["runner"]["required"] is True
    assert inputs["compute"]["required"] is True
    assert inputs["environment_name"]["required"] is True
    assert inputs["environment_version"]["required"] is True
    assert inputs["instance_type"]["required"] is True
    assert "ubuntu-24.04" not in workflow
    assert "Standard_DS2_v2" not in workflow
    assert "--compute \"$COMPUTE\"" in workflow
    assert "--environment_name \"$ENVIRONMENT_NAME\"" in workflow
    assert "--environment_version \"$ENVIRONMENT_VERSION\"" in workflow
    action_refs = re.findall(r"uses:\s+\S+@(\S+)", workflow)
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)


def test_online_yaml_assets_are_valid():
    repository_root = Path(__file__).parents[1]
    paths = [
        repository_root / ".github/workflows/python-sdk-v2-online.yml",
        repository_root / "templates/python-sdk-v2/create-online-endpoint.yml",
        repository_root / "templates/python-sdk-v2/create-online-deployment.yml",
        repository_root / "templates/python-sdk-v2/test-online-endpoint.yml",
    ]

    for path in paths:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None


def test_online_endpoint_uses_attached_arc_kubernetes_compute(monkeypatch):
    client = Mock()
    compute = _arc_kubernetes_compute()
    client.compute.get.return_value = compute
    endpoint_poller = Mock()
    endpoint_poller.result.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    client.online_endpoints.begin_create_or_update.return_value = endpoint_poller
    client.online_endpoints.get.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    endpoint_type = Mock(return_value="kubernetes-endpoint")
    monkeypatch.setattr(create_online_endpoint, "create_ml_client", lambda _: client)
    monkeypatch.setattr(
        create_online_endpoint,
        "KubernetesOnlineEndpoint",
        endpoint_type,
    )

    result = create_online_endpoint.run(_online_endpoint_args())

    assert result.provisioning_state == "Succeeded"
    client.compute.get.assert_called_once_with("arc-inference")
    endpoint_type.assert_called_once_with(
        name="endpoint",
        description=None,
        auth_mode="aml_token",
        compute=compute.id,
    )
    client.online_endpoints.begin_create_or_update.assert_called_once_with(
        "kubernetes-endpoint"
    )


def test_online_compute_accepts_direct_aks_attachment_with_uami():
    client = Mock()
    compute = _arc_kubernetes_compute()
    compute.resource_id = (
        "/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.ContainerService/managedClusters/private-aks"
    )
    client.compute.get.return_value = compute

    result = aml_client.get_kubernetes_online_compute(client, "direct-aks")

    assert result is compute


def test_online_compute_rejects_unsupported_cluster_resource():
    client = Mock()
    compute = _arc_kubernetes_compute()
    compute.resource_id = (
        "/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.Compute/virtualMachines/not-kubernetes"
    )
    client.compute.get.return_value = compute

    with pytest.raises(RuntimeError, match="direct AKS managedClusters"):
        aml_client.get_kubernetes_online_compute(client, "not-kubernetes")


def test_online_compute_requires_dedicated_namespace_and_uami():
    client = Mock()
    compute = _arc_kubernetes_compute()
    compute.namespace = "default"
    client.compute.get.return_value = compute

    with pytest.raises(RuntimeError, match="dedicated non-default namespace"):
        aml_client.get_kubernetes_online_compute(client, "arc-inference")

    compute.namespace = "azureml-inference"
    compute.identity = SimpleNamespace(
        type="system_assigned",
        user_assigned_identities=None,
    )
    with pytest.raises(RuntimeError, match="user-assigned managed identity"):
        aml_client.get_kubernetes_online_compute(client, "arc-inference")


def test_prebuilt_environment_contract_requires_digest_and_no_build():
    client = Mock()
    client.environments.get.return_value = SimpleNamespace(
        id="azureml:/environments/inference/versions/7",
        image="private.azurecr.io/inference@sha256:" + ("a" * 64),
        build=None,
        conda_file=None,
    )

    result = aml_client.get_prebuilt_environment(client, "inference", "7")

    assert result.id == "azureml:/environments/inference/versions/7"
    client.environments.get.assert_called_once_with(
        name="inference",
        version="7",
    )


@pytest.mark.parametrize(
    ("image", "build", "conda_file", "message"),
    [
        (
            "private.azurecr.io/inference:latest",
            None,
            None,
            "sha256 digest",
        ),
        (
            "private.azurecr.io/inference@sha256:" + ("a" * 64),
            SimpleNamespace(path="."),
            None,
            "image-only environment",
        ),
        (
            "private.azurecr.io/inference@sha256:" + ("a" * 64),
            None,
            "conda.yaml",
            "image-only environment",
        ),
    ],
)
def test_prebuilt_environment_rejects_mutable_or_buildable_assets(
    image,
    build,
    conda_file,
    message,
):
    client = Mock()
    client.environments.get.return_value = SimpleNamespace(
        image=image,
        build=build,
        conda_file=conda_file,
    )

    with pytest.raises(RuntimeError, match=message):
        aml_client.get_prebuilt_environment(client, "inference", "7")


def test_online_deployment_uses_kubernetes_and_exact_environment(monkeypatch):
    client = Mock()
    client.models.get.return_value = SimpleNamespace(
        id="azureml:model:1",
        type="mlflow_model",
    )
    environment = SimpleNamespace(
        id="azureml:/environments/inference/versions/7",
        image="private.azurecr.io/inference@sha256:" + ("a" * 64),
        build=None,
        conda_file=None,
    )
    client.environments.get.return_value = environment
    deployment_poller = Mock()
    deployment_poller.result.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    client.online_deployments.begin_create_or_update.return_value = deployment_poller
    client.online_deployments.get.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    endpoint = SimpleNamespace(traffic={}, provisioning_state="Succeeded")
    client.online_endpoints.get.return_value = endpoint
    endpoint_poller = Mock()
    endpoint_poller.result.return_value = endpoint
    client.online_endpoints.begin_create_or_update.return_value = endpoint_poller
    deployment_type = Mock(return_value="kubernetes-deployment")
    monkeypatch.setattr(create_online_deployment, "create_ml_client", lambda _: client)
    monkeypatch.setattr(
        create_online_deployment,
        "KubernetesOnlineDeployment",
        deployment_type,
    )

    create_online_deployment.run(_online_deployment_args())

    assert endpoint.traffic == {"blue": 100}
    deployment_type.assert_called_once_with(
        name="blue",
        endpoint_name="endpoint",
        model="azureml:model:1",
        environment=environment.id,
        instance_type="cpu-small",
        instance_count=2,
    )
    client.online_deployments.begin_create_or_update.assert_called_once_with(
        "kubernetes-deployment"
    )
    client.online_endpoints.begin_create_or_update.assert_called_once_with(endpoint)


def test_online_deployment_repeat_update_waits_before_traffic(monkeypatch):
    events = []
    client = Mock()
    client.models.get.return_value = SimpleNamespace(
        id="azureml:model:1",
        type="mlflow_model",
    )
    client.environments.get.return_value = SimpleNamespace(
        id="azureml:/environments/inference/versions/7",
        image="private.azurecr.io/inference@sha256:" + ("a" * 64),
        build=None,
        conda_file=None,
    )
    successful_poller = Mock()
    successful_poller.result.side_effect = lambda: (
        events.append("deployment-result")
        or SimpleNamespace(provisioning_state="Succeeded")
    )
    client.online_deployments.begin_create_or_update.side_effect = [
        ResourceExistsError("operation already in progress"),
        successful_poller,
    ]
    client.online_deployments.get.side_effect = [
        SimpleNamespace(provisioning_state="Updating"),
        SimpleNamespace(provisioning_state="Succeeded"),
        SimpleNamespace(provisioning_state="Succeeded"),
    ]
    endpoint = SimpleNamespace(traffic={}, provisioning_state="Succeeded")
    client.online_endpoints.get.return_value = endpoint
    endpoint_poller = Mock()
    endpoint_poller.result.side_effect = lambda: (
        events.append("traffic-result") or endpoint
    )
    client.online_endpoints.begin_create_or_update.return_value = endpoint_poller
    monkeypatch.setattr(create_online_deployment, "create_ml_client", lambda _: client)
    monkeypatch.setattr(aml_client.time, "sleep", lambda _: events.append("state-check"))

    create_online_deployment.run(_online_deployment_args())

    assert client.online_deployments.begin_create_or_update.call_count == 2
    assert events == ["state-check", "deployment-result", "traffic-result"]
    assert endpoint.traffic == {"blue": 100}


def test_online_endpoint_completes_before_deployment_begins(monkeypatch):
    events = []
    client = Mock()
    client.compute.get.return_value = _arc_kubernetes_compute()
    client.models.get.return_value = SimpleNamespace(
        id="azureml:model:1",
        type="mlflow_model",
    )
    client.environments.get.return_value = SimpleNamespace(
        id="azureml:/environments/inference/versions/7",
        image="private.azurecr.io/inference@sha256:" + ("a" * 64),
        build=None,
        conda_file=None,
    )
    endpoint_poller = Mock()
    endpoint_poller.result.side_effect = lambda: (
        events.append("endpoint-result")
        or SimpleNamespace(provisioning_state="Succeeded")
    )
    endpoint = SimpleNamespace(traffic={}, provisioning_state="Succeeded")
    client.online_endpoints.begin_create_or_update.side_effect = [
        endpoint_poller,
        Mock(result=Mock(return_value=endpoint)),
    ]
    client.online_endpoints.get.return_value = endpoint
    deployment_poller = Mock()
    deployment_poller.result.side_effect = lambda: (
        events.append("deployment-result")
        or SimpleNamespace(provisioning_state="Succeeded")
    )
    client.online_deployments.begin_create_or_update.side_effect = lambda _: (
        events.append("deployment-begin") or deployment_poller
    )
    client.online_deployments.get.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    monkeypatch.setattr(create_online_endpoint, "create_ml_client", lambda _: client)
    monkeypatch.setattr(create_online_deployment, "create_ml_client", lambda _: client)

    create_online_endpoint.run(_online_endpoint_args())
    create_online_deployment.run(_online_deployment_args())

    assert events.index("endpoint-result") < events.index("deployment-begin")


def test_missing_registered_model_has_actionable_error():
    client = Mock()
    client.models.get.side_effect = RuntimeError("not found")

    with pytest.raises(RuntimeError, match="training/model registration workflow"):
        aml_client.get_registered_model(client, "model", "7")


def test_no_code_deployment_rejects_non_mlflow_model():
    client = Mock()
    client.models.get.return_value = SimpleNamespace(type="custom_model")

    with pytest.raises(RuntimeError, match="require an MLflow model"):
        aml_client.get_registered_model(
            client,
            "model",
            "7",
            require_mlflow=True,
        )


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


def test_batch_endpoint_poller_completes_before_deployment_begin(monkeypatch):
    events = []
    client = Mock()
    endpoint_poller = Mock()
    endpoint_poller.result.side_effect = lambda: (
        events.append("endpoint-result")
        or SimpleNamespace(provisioning_state="Succeeded")
    )
    client.batch_endpoints.begin_create_or_update.side_effect = lambda _: (
        events.append("endpoint-begin") or endpoint_poller
    )
    client.batch_endpoints.get.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    client.models.get.return_value = SimpleNamespace(
        id="azureml:model:1",
        type="mlflow_model",
    )
    deployment_poller = Mock()
    deployment_poller.result.side_effect = lambda: (
        events.append("deployment-result")
        or SimpleNamespace(provisioning_state="Succeeded")
    )
    client.batch_deployments.begin_create_or_update.side_effect = lambda _: (
        events.append("deployment-begin") or deployment_poller
    )
    client.batch_deployments.get.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    endpoint = SimpleNamespace(
        defaults=SimpleNamespace(deployment_name=None),
        provisioning_state="Succeeded",
    )
    client.batch_endpoints.get.return_value = endpoint
    monkeypatch.setattr(create_batch_endpoint, "create_ml_client", lambda _: client)
    monkeypatch.setattr(create_batch_deployment, "create_ml_client", lambda _: client)

    create_batch_endpoint.run(_batch_endpoint_args())
    create_batch_deployment.run(_batch_deployment_args())

    assert events.index("endpoint-result") < events.index("deployment-begin")


def test_batch_deployment_poller_completes_before_invocation(monkeypatch):
    events = []
    client = Mock()
    client.models.get.return_value = SimpleNamespace(
        id="azureml:model:1",
        type="mlflow_model",
    )
    deployment_poller = Mock()
    deployment_poller.result.side_effect = lambda: (
        events.append("deployment-result")
        or SimpleNamespace(provisioning_state="Succeeded")
    )
    client.batch_deployments.begin_create_or_update.return_value = deployment_poller
    client.batch_deployments.get.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    endpoint = SimpleNamespace(
        defaults=SimpleNamespace(deployment_name=None),
        provisioning_state="Succeeded",
    )
    client.batch_endpoints.get.return_value = endpoint
    endpoint_poller = Mock()
    endpoint_poller.result.return_value = endpoint
    client.batch_endpoints.begin_create_or_update.return_value = endpoint_poller
    client.batch_endpoints.invoke.side_effect = lambda **_: (
        events.append("invoke") or SimpleNamespace(name="batch-job")
    )
    monkeypatch.setattr(create_batch_deployment, "create_ml_client", lambda _: client)
    monkeypatch.setattr(test_batch_endpoint, "create_ml_client", lambda _: client)
    monkeypatch.setattr(
        test_batch_endpoint,
        "wait_for_job",
        lambda *_: SimpleNamespace(status="Completed"),
    )

    create_batch_deployment.run(_batch_deployment_args())
    test_batch_endpoint.run(_batch_invoke_args())

    assert events.index("deployment-result") < events.index("invoke")


def test_batch_endpoint_repeat_update_waits_then_retries(monkeypatch):
    events = []
    client = Mock()
    successful_poller = Mock()
    successful_poller.result.side_effect = lambda: (
        events.append("update-result")
        or SimpleNamespace(provisioning_state="Succeeded")
    )
    client.batch_endpoints.begin_create_or_update.side_effect = [
        ResourceExistsError("operation already in progress"),
        successful_poller,
    ]
    client.batch_endpoints.get.side_effect = [
        SimpleNamespace(provisioning_state="Updating"),
        SimpleNamespace(provisioning_state="Succeeded"),
        SimpleNamespace(provisioning_state="Succeeded"),
    ]
    monkeypatch.setattr(create_batch_endpoint, "create_ml_client", lambda _: client)
    monkeypatch.setattr(aml_client.time, "sleep", lambda _: events.append("state-check"))

    result = create_batch_endpoint.run(_batch_endpoint_args())

    assert result.provisioning_state == "Succeeded"
    assert client.batch_endpoints.begin_create_or_update.call_count == 2
    assert events == ["state-check", "update-result"]


def test_batch_endpoint_terminal_operation_failure_propagates(monkeypatch):
    client = Mock()
    poller = Mock()
    poller.result.return_value = SimpleNamespace(provisioning_state="Succeeded")
    client.batch_endpoints.begin_create_or_update.return_value = poller
    client.batch_endpoints.get.return_value = SimpleNamespace(
        provisioning_state="Failed"
    )
    monkeypatch.setattr(create_batch_endpoint, "create_ml_client", lambda _: client)

    with pytest.raises(RuntimeError, match="provisioning finished with state Failed"):
        create_batch_endpoint.run(_batch_endpoint_args())

    client.batch_endpoints.get.assert_called_once_with("batch-endpoint")


def test_batch_deployment_uses_explicit_immutable_environment(monkeypatch):
    client = Mock()
    client.models.get.return_value = SimpleNamespace(
        id="azureml:model:1",
        type="mlflow_model",
    )
    deployment_poller = Mock()
    deployment_poller.result.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    client.batch_deployments.begin_create_or_update.return_value = deployment_poller
    client.batch_deployments.get.return_value = SimpleNamespace(
        provisioning_state="Succeeded"
    )
    endpoint = SimpleNamespace(
        defaults=SimpleNamespace(deployment_name=None),
        provisioning_state="Succeeded",
    )
    client.batch_endpoints.get.return_value = endpoint
    endpoint_poller = Mock()
    endpoint_poller.result.return_value = endpoint
    client.batch_endpoints.begin_create_or_update.return_value = endpoint_poller
    batch_deployment_type = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(create_batch_deployment, "create_ml_client", lambda _: client)
    monkeypatch.setattr(
        create_batch_deployment,
        "BatchDeployment",
        batch_deployment_type,
    )

    create_batch_deployment.run(_batch_deployment_args())

    assert (
        batch_deployment_type.call_args.kwargs["environment"]
        == create_batch_deployment.DEFAULT_BATCH_ENVIRONMENT
    )
    assert "image" not in batch_deployment_type.call_args.kwargs
    assert "code_configuration" not in batch_deployment_type.call_args.kwargs


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "latest",
        "azureml://registries/azureml/environments/sklearn-1.5",
        "azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
        "azureml://registries/azureml/environments/sklearn-1.5/versions/latest",
        "azureml:sklearn-1.5@latest",
        "azureml:sklearn-1.5:latest",
        "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        "conda.yml",
    ],
)
def test_batch_deployment_rejects_mutable_environment_reference(
    monkeypatch,
    reference,
):
    create_client = Mock()
    monkeypatch.setattr(create_batch_deployment, "create_ml_client", create_client)
    args = _batch_deployment_args()
    args.environment = reference

    with pytest.raises(argparse.ArgumentTypeError, match="immutable|numeric version"):
        create_batch_deployment.run(args)

    create_client.assert_not_called()


def test_batch_workflow_uses_pinned_curated_environment():
    workflow = (
        Path(__file__).parents[1]
        / ".github"
        / "workflows"
        / "python-sdk-v2-batch.yml"
    ).read_text(encoding="utf-8")

    assert (
        "default: "
        "azureml://registries/azureml/environments/sklearn-1.5/versions/53"
    ) in workflow
    assert "DEPLOYMENT_ENVIRONMENT: ${{ inputs.deployment_environment }}" in workflow
    assert '--environment "$DEPLOYMENT_ENVIRONMENT"' in workflow


def test_batch_cli_defaults_to_immutable_prebuilt_environment(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_batch_deployment.py",
            "--subscription_id",
            "sub",
            "--resource_group",
            "rg",
            "--workspace_name",
            "ws",
            "--deployment_name",
            "batch-deployment",
            "--endpoint_name",
            "batch-endpoint",
            "--model_name",
            "model",
            "--model_version",
            "1",
            "--compute",
            "batch-compute",
        ],
    )

    args = create_batch_deployment.parse_args()

    assert args.environment == create_batch_deployment.DEFAULT_BATCH_ENVIRONMENT


def _batch_endpoint_args():
    return argparse.Namespace(
        endpoint_name="batch-endpoint",
        description=None,
        auth_mode="aad_token",
    )


def _arc_kubernetes_compute():
    return SimpleNamespace(
        id="azureml:/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws/computes/arc-inference",
        type="Kubernetes",
        resource_id="/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.Kubernetes/connectedClusters/private-inference",
        provisioning_state="Succeeded",
        namespace="azureml-inference",
        identity=SimpleNamespace(
            type="user_assigned",
            user_assigned_identities=[
                SimpleNamespace(resource_id="/subscriptions/sub/uami")
            ],
        ),
    )


def _online_endpoint_args():
    return argparse.Namespace(
        endpoint_name="endpoint",
        compute="arc-inference",
        description=None,
        auth_mode="aml_token",
    )


def _online_deployment_args():
    return argparse.Namespace(
        deployment_name="blue",
        endpoint_name="endpoint",
        model_name="model",
        model_version="1",
        environment_name="inference",
        environment_version="7",
        instance_type="cpu-small",
        instance_count=2,
        traffic_allocation=100,
    )


def _batch_deployment_args():
    return argparse.Namespace(
        deployment_name="batch-deployment",
        description=None,
        endpoint_name="batch-endpoint",
        model_name="model",
        model_version="1",
        compute="batch-compute",
        environment=create_batch_deployment.DEFAULT_BATCH_ENVIRONMENT,
        instance_count=1,
        max_concurrency_per_instance=1,
        mini_batch_size=10,
        output_file_name="predictions.csv",
    )


def _batch_invoke_args():
    return argparse.Namespace(
        endpoint_name="batch-endpoint",
        request_batch_file="azureml://datastores/test/paths/input",
        request_type="uri_folder",
    )


def test_failed_job_downloads_child_logs_and_surfaces_root_cause(
    monkeypatch,
    tmp_path,
    capsys,
):
    failed = SimpleNamespace(name="parent", status="Failed", error=None)
    child = SimpleNamespace(name="child", status="Failed", error=None)
    client = Mock()
    client.jobs.get.return_value = failed
    client.jobs.list.return_value = [child]

    def download_child_logs(*, name, download_path, all):
        assert name == "child"
        assert all is False
        log_path = Path(download_path) / "user_logs" / "std_log.txt"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(
            "starting training\n"
            "Traceback (most recent call last):\n"
            "  File \"train.py\", line 42, in <module>\n"
            "ValueError: decisive root cause\n",
            encoding="utf-8",
        )

    client.jobs.download.side_effect = download_child_logs
    monkeypatch.setattr(aml_client.time, "sleep", Mock())
    monkeypatch.setenv("AML_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))

    with pytest.raises(RuntimeError, match="finished with status Failed"):
        aml_client.wait_for_job(client, "parent", poll_interval_seconds=0)

    output = capsys.readouterr().out
    assert "Child job child: status=Failed" in output
    assert "Traceback (most recent call last)" in output
    assert "ValueError: decisive root cause" in output
    client.jobs.download.assert_called_once_with(
        name="child",
        download_path=str(tmp_path / "diagnostics" / "parent" / "child"),
        all=False,
    )


def test_failed_job_reports_download_failure_and_falls_back_to_parent(
    monkeypatch,
    tmp_path,
    capsys,
):
    failed = SimpleNamespace(name="parent", status="Failed", error=None)
    client = Mock()
    client.jobs.get.return_value = failed
    client.jobs.list.return_value = []
    client.jobs.download.side_effect = RuntimeError("private storage unavailable")
    monkeypatch.setenv("AML_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))

    with pytest.raises(RuntimeError, match="finished with status Failed"):
        aml_client.wait_for_job(client, "parent", poll_interval_seconds=0)

    output = capsys.readouterr().out
    assert (
        "Unable to download diagnostics for job parent: "
        "private storage unavailable"
    ) in output
    client.jobs.download.assert_called_once_with(
        name="parent",
        download_path=str(tmp_path / "diagnostics" / "parent" / "parent"),
        all=False,
    )


def test_failed_leaf_retries_full_download_when_standard_download_is_pointer(
    monkeypatch,
    tmp_path,
    capsys,
):
    failed_leaf = SimpleNamespace(
        name="imgbldrun_1175c66",
        status="Failed",
        error=None,
    )
    client = Mock()
    client.jobs.get.return_value = failed_leaf
    client.jobs.list.return_value = []

    def download_leaf_logs(*, name, download_path, all):
        assert name == "imgbldrun_1175c66"
        download_root = Path(download_path)
        if not all:
            (download_root / "artifact_download_info.json").write_text(
                '{"status": "available"}',
                encoding="utf-8",
            )
            return
        log_path = download_root / "logs" / "image_build.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(
            "Traceback (most recent call last):\n"
            "RuntimeError: image build root cause\n",
            encoding="utf-8",
        )

    client.jobs.download.side_effect = download_leaf_logs
    monkeypatch.setenv("AML_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))

    with pytest.raises(RuntimeError, match="finished with status Failed"):
        aml_client.wait_for_job(
            client,
            "imgbldrun_1175c66",
            poll_interval_seconds=0,
        )

    output = capsys.readouterr().out
    assert "retrying full job download" in output
    assert "RuntimeError: image build root cause" in output
    download_path = str(
        tmp_path
        / "diagnostics"
        / "imgbldrun_1175c66"
        / "imgbldrun_1175c66"
    )
    assert client.jobs.download.call_args_list == [
        call(
            name="imgbldrun_1175c66",
            download_path=download_path,
            all=False,
        ),
        call(
            name="imgbldrun_1175c66",
            download_path=download_path,
            all=True,
        ),
    ]


def test_diagnostic_read_failure_is_explicit(monkeypatch, tmp_path, capsys):
    job = SimpleNamespace(name="parent", status="Failed", error=None)
    client = Mock()

    def download_logs(*, name, download_path, all):
        log_path = Path(download_path) / "std_log.txt"
        log_path.write_text("Error: hidden", encoding="utf-8")

    client.jobs.download.side_effect = download_logs
    original_open = Path.open

    def fail_diagnostic_read(path, *args, **kwargs):
        if path.name == "std_log.txt" and args == ("rb",):
            raise OSError("read denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_diagnostic_read)

    aml_client._download_and_print_job_diagnostics(client, job, tmp_path)

    output = capsys.readouterr().out
    assert "Unable to read diagnostic file" in output
    assert "read denied" in output


def test_diagnostic_output_is_redacted_and_bounded():
    sensitive_value = "sample-sensitive-value"
    content = (
        "authorization="
        + "Bearer "
        + sensitive_value
        + "\napi_key="
        + sensitive_value
        + "\nError: request failed?sig="
        + sensitive_value
        + "&se=tomorrow\n"
        + ("x" * 200)
    )

    excerpt = aml_client._diagnostic_excerpt(content, max_chars=80)

    assert sensitive_value not in excerpt
    assert "[REDACTED]" in excerpt
    assert len(excerpt) < 130
    assert "diagnostic output truncated" in excerpt


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
        model_type="mlflow_model",
    )

    train_and_register_model.run(args)

    registered = client.models.create_or_update.call_args.args[0]
    assert registered.name == "forecast"
    assert registered.path == "azureml://jobs/training-job/outputs/model/paths/"
    assert registered.type == "mlflow_model"


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
        model_type="mlflow_model",
    )

    train_and_register_model.run(args)

    assert output_file.read_text(encoding="utf-8").splitlines() == [
        "training_job_name=training-job",
        "model_name=forecast",
        "model_version=3",
    ]


def test_training_job_name_is_written_before_failed_wait(monkeypatch, tmp_path):
    client = Mock()
    client.jobs.create_or_update.return_value = SimpleNamespace(name="training-job")
    output_file = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(train_and_register_model, "create_ml_client", lambda _: client)
    monkeypatch.setattr(train_and_register_model, "load_job", lambda source: source)
    monkeypatch.setattr(
        train_and_register_model,
        "wait_for_job",
        Mock(side_effect=RuntimeError("training failed")),
    )
    args = argparse.Namespace(
        job_file="jobs/train.yml",
        model_name="forecast",
        model_output_name="model",
        model_type="mlflow_model",
    )

    with pytest.raises(RuntimeError, match="training failed"):
        train_and_register_model.run(args)

    assert output_file.read_text(encoding="utf-8").splitlines() == [
        "training_job_name=training-job"
    ]


def test_train_workflow_uploads_failed_diagnostics_with_pinned_action():
    workflow = (
        Path(__file__).parents[1]
        / ".github"
        / "workflows"
        / "python-sdk-v2-train-register.yml"
    ).read_text(encoding="utf-8")

    assert "if: ${{ failure() }}" in workflow
    assert (
        "uses: actions/upload-artifact@"
        "ea165f8d65b6e75b540449e92b4886f43607fa02"
    ) in workflow
    assert "path: aml-diagnostics" in workflow
    assert "if-no-files-found: warn" in workflow
