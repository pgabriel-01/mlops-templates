# Azure ML Python SDK v2 reusable workflows

These GitHub-native workflows use Azure ML Python SDK v2 for ML operations. They
authenticate with GitHub Environment OIDC through `azure/login`, then construct
`MLClient` with `DefaultAzureCredential` and explicit workspace coordinates.
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
`model_version`. The online workflow requires `endpoint_name`,
`deployment_name`, `model_name`, `model_version`, and `request_file`, and outputs
the endpoint and deployment names. The batch workflow requires those endpoint
and model inputs plus `compute`, `request_batch_file`, and optionally
`request_type`, and also outputs the endpoint and deployment names. Each accepts
an optional `runner` label (default `ubuntu-24.04`).

Both deployment workflows create or update resources idempotently, wait for
long-running operations, and test the deployment. Batch invocation waits for a
terminal job state and fails the workflow with parent and child-job diagnostics.

Promote to `test` or `prod` by calling the same reusable workflow with a
different GitHub Environment and environment-scoped OIDC secrets and variables.
