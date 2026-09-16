# Azure ML Python SDK v2 reusable workflows

These GitHub-native workflows use Azure ML Python SDK v2 for ML operations. They
authenticate with GitHub Environment OIDC through `azure/login`, then use
`AzureCliCredential` to consume that exact signed-in identity and construct
`MLClient` with explicit workspace coordinates. The reusable workflows set
`AZUREML_CREDENTIAL_MODE=azure-cli`; token acquisition is validated before any
Azure ML operation, and CI never falls back to a runner node's managed identity.
They never use `az ml` or interactive browser authentication.

Call the workflows with an immutable `sdk_ref`:

```yaml
jobs:
  train-dev:
    uses: OWNER/SDK_REPOSITORY/.github/workflows/python-sdk-v2-train-register.yml@COMMIT_SHA
    with:
      environment: dev
      sdk_repository: OWNER/SDK_REPOSITORY
      sdk_ref: COMMIT_SHA
      resource_group: ${{ vars.AZURE_RESOURCE_GROUP }}
      workspace_name: ${{ vars.AZURE_ML_WORKSPACE }}
      job_file: jobs/train.yml
      model_name: ${{ vars.MODEL_NAME }}
    secrets:
      AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
      AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
      AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
```

The training workflow outputs `training_job_name`, `model_name`, and
`model_version`. Its `model_output_name` defaults to `model` and its
`model_type` defaults to `mlflow_model`. The batch workflow requires its
endpoint and model inputs plus `compute`, `request_batch_file`, and optionally
`request_type`, and outputs the endpoint and deployment names.

The online workflow targets an Azure ML **Kubernetes online endpoint**, not a
managed online endpoint. Its required serving inputs are:

- `compute`: an existing Azure ML Kubernetes compute backed by an Azure
  Arc-enabled `Microsoft.Kubernetes/connectedClusters` resource
- `environment_name` and `environment_version`: an existing, versioned Azure ML
  environment whose only runtime source is a prebuilt image pinned by
  `@sha256:<digest>`
- `instance_type`: an Azure ML Kubernetes instance type defined by the cluster
  administrator
- `runner`: the private ARC runner label with network access to the private
  workspace and inference endpoint
- `endpoint_name`, `deployment_name`, `model_name`, `model_version`, and
  `request_file`

`instance_count` and `traffic_allocation` remain optional. The workflow outputs
the endpoint and deployment names.

The online compute is an infrastructure prerequisite and is validated before
endpoint creation. It must be in `Succeeded` state, use a dedicated non-default
namespace, and have a user-assigned managed identity. The compute identity
pulls the prebuilt image and accesses required Azure resources; the workflow
never reads AKS credentials or uses the AKS node identity.

Do not attach a local-accounts-disabled AKS cluster directly to Azure ML. The
Azure ML extension does not support that direct attachment mode. Use an
Azure Arc-enabled Kubernetes attachment or a separate supported secure cluster;
do not enable AKS local accounts to make deployment work.

Infrastructure must install the Azure ML extension with inference enabled and
HTTPS preserved. At minimum, configure `enableInference=true`,
`allowInsecureConnections=false`, `sslSecret`, and `sslCname`, plus the
inference router service type. Infrastructure also owns the dedicated
namespace, user node pool, workload identity/UAMI, TLS secret, and Kubernetes
instance types. None of those credentials or Kubernetes administration details
are workflow inputs.

Outside CI, the scripts use `DefaultAzureCredential` for local development with
interactive browser and managed identity credentials excluded. Sign in with
`az login` or configure another supported local credential. Set
`AZUREML_CREDENTIAL_MODE=azure-cli` to require an existing Azure CLI login
locally; `AZUREML_CREDENTIAL_MODE=default` is rejected in CI.

Both deployment workflows create or update resources idempotently, wait for
long-running operations, and test the deployment. Batch invocation waits for a
terminal job state and fails the workflow with parent and child-job diagnostics.
The deployment workflows intentionally use Azure ML's MLflow no-code deployment
path and fail with an actionable error if the referenced model is not MLflow.
For online serving, the explicit registered environment prevents Azure ML from
creating an anonymous environment or starting a workspace image build. This is
required when workspace storage enforces `allowSharedKeyAccess=false`.

## Consumer migration

Replace managed-online inputs with the infrastructure outputs and keep the
workflow pinned to an immutable commit:

```yaml
jobs:
  deploy-online:
    uses: OWNER/mlops-templates/.github/workflows/python-sdk-v2-online.yml@COMMIT_SHA
    with:
      environment: prod
      sdk_repository: OWNER/mlops-templates
      sdk_ref: COMMIT_SHA
      resource_group: ${{ vars.AZURE_RESOURCE_GROUP }}
      workspace_name: ${{ vars.AZURE_ML_WORKSPACE }}
      endpoint_name: ${{ vars.ONLINE_ENDPOINT_NAME }}
      deployment_name: blue
      model_name: ${{ needs.train.outputs.model_name }}
      model_version: ${{ needs.train.outputs.model_version }}
      compute: ${{ vars.ONLINE_COMPUTE }}
      environment_name: ${{ vars.ONLINE_ENVIRONMENT_NAME }}
      environment_version: ${{ vars.ONLINE_ENVIRONMENT_VERSION }}
      instance_type: ${{ vars.ONLINE_INSTANCE_TYPE }}
      instance_count: 1
      request_file: samples/online-request.json
      runner: ${{ vars.PRIVATE_ARC_RUNNER }}
    secrets:
      AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
      AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
      AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
```

The coordinated infrastructure configuration keys are `online_compute`,
`online_environment_name`, `online_environment_version`, and
`online_instance_type`. Map them to the uppercase repository/environment
variables shown above. Remove any managed-online VM SKU such as
`Standard_DS2_v2`; Kubernetes `instance_type` is a cluster-defined resource
profile. Keep the private ARC runner and existing OIDC secrets unchanged.

Promote to `test` or `prod` by calling the same reusable workflow with a
different GitHub Environment and environment-scoped OIDC secrets and variables.
