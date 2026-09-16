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
`model_type` defaults to `mlflow_model`. The online workflow requires `endpoint_name`,
`deployment_name`, `model_name`, `model_version`, and `request_file`, and outputs
the endpoint and deployment names. The batch workflow requires those endpoint
and model inputs plus `compute`, `request_batch_file`, and optionally
`request_type`, and also outputs the endpoint and deployment names. Its
`deployment_environment` defaults to the immutable curated environment
`azureml://registries/azureml/environments/sklearn-1.5/versions/53`. Override it
only with another versioned Azure ML environment reference. Supplying an
explicit prebuilt environment prevents Azure ML from generating an anonymous
Conda environment and workspace image build, which is incompatible with
workspaces that enforce `allowSharedKeyAccess=false`. Each workflow accepts an
optional `runner` label (default `ubuntu-24.04`).

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
Batch deployment also rejects mutable or unversioned environment references.

Promote to `test` or `prod` by calling the same reusable workflow with a
different GitHub Environment and environment-scoped OIDC secrets and variables.
