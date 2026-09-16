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
`request_type`, and outputs the endpoint and deployment names. Its
`deployment_environment` defaults to the immutable curated environment
`azureml://registries/azureml/environments/sklearn-1.5/versions/53`. Override it
only with another versioned Azure ML environment reference. Supplying an
explicit prebuilt environment prevents Azure ML from generating an anonymous
Conda environment and workspace image build, which is incompatible with
workspaces that enforce `allowSharedKeyAccess=false`. Registry shorthand is
resolved with a registry-scoped SDK client using the same OIDC-backed
credential. Because the SDK returns registry shorthand from `Environment.id`,
the exact numeric version is verified by lookup and its full ARM resource ID is
constructed from the registry client's authoritative operation scope. Missing
or unsafe scope data fails closed, and the batch service never receives
registry shorthand or an `Environment` object in
`ModelConfiguration.EnvironmentId`.

The online workflow targets an Azure ML **Kubernetes online endpoint**, not a
managed online endpoint. It supports two explicit deployment modes:

- The default image-only mode requires `environment_name` and
  `environment_version` for an existing Azure ML environment whose only runtime
  source is a prebuilt image pinned by `@sha256:<digest>`.
- Set `mlflow_no_code: true` to use Azure ML's supported MLflow online no-code
  deployment. In this mode, omit both environment inputs. The deployment omits
  the environment and scoring code configuration so Azure ML supplies its
  curated inference base and installs the dependencies declared by the
  registered MLflow model at container runtime.

The shared required serving inputs are:

- `compute`: preferably an existing direct private AKS
  `Microsoft.ContainerService/managedClusters` compute attachment; an Azure
  Arc-enabled `Microsoft.Kubernetes/connectedClusters` attachment is the
  fallback
- `instance_type`: an Azure ML Kubernetes instance type defined by the cluster
  administrator
- `runner`: the private ARC runner label with network access to the private
  workspace and inference endpoint
- `tls_ca_key_vault_secret_id`: the full Key Vault secret resource ID for a
  secret whose value is the PEM CA certificate bundle that signed the private
  inference ingress certificate
- `endpoint_name`, `deployment_name`, `model_name`, `model_version`, and
  `request_file`

`environment_name`, `environment_version`, `mlflow_no_code`, `instance_count`,
`traffic_allocation`, and `endpoint_uami_resource_id` are optional workflow
inputs. The environment pair is conditionally required when `mlflow_no_code`
is false, and supplying either environment input in no-code mode fails before
Azure access. When supplied, `endpoint_uami_resource_id` must be the full
resource ID of a user-assigned managed identity; system-assigned and AKS node
identity fallbacks are rejected. The pinned SDK supports an explicit identity
on `KubernetesOnlineEndpoint`. When this optional input is omitted, the
workflow does not force an endpoint identity: the required UAMI on the attached
Azure ML Kubernetes compute remains the serving identity used for image pulls
and Azure resource access. The workflow outputs the endpoint and deployment
names.

The online compute is an infrastructure prerequisite and is validated before
endpoint creation. It must be in `Succeeded` state, use a dedicated non-default
namespace, and have a user-assigned managed identity. The compute identity
pulls the prebuilt image and accesses required Azure resources; the workflow
never reads AKS credentials or uses the AKS node identity.

For direct AKS with local accounts disabled, infrastructure must create the
per-workspace AKS Trusted Access role binding for
`Microsoft.MachineLearningServices/workspaces/mlworkload` before attaching the
compute. This direct `managedClusters` attachment is supported in Azure public
cloud and does not require Azure Arc. Do not enable AKS local accounts. Azure
Arc-enabled `connectedClusters` remains a supported alternative.

Infrastructure must install the Azure ML extension with inference enabled and
HTTPS preserved. At minimum, configure `enableInference=true`,
`allowInsecureConnections=false`, `sslSecret`, and `sslCname`, plus the
inference router with an internal ingress service type. Infrastructure also
owns the dedicated namespace, user node pool, workload identity/UAMI, TLS
secret, and Kubernetes instance types. None of those credentials or Kubernetes
administration details are workflow inputs.

The OIDC identity must have permission to read the specified Key Vault secret.
After `azure/login`, the workflow retrieves the secret value without printing
it, writes it to a mode `0600` temporary file, validates it as a PEM certificate
bundle, and exposes it as `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` only to the
endpoint invocation step. TLS verification remains enabled. A missing,
unreadable, empty, or invalid CA bundle fails the job; an `always()` cleanup
step deletes the temporary file.

Outside CI, the scripts use `DefaultAzureCredential` for local development with
interactive browser and managed identity credentials excluded. Sign in with
`az login` or configure another supported local credential. Set
`AZUREML_CREDENTIAL_MODE=azure-cli` to require an existing Azure CLI login
locally; `AZUREML_CREDENTIAL_MODE=default` is rejected in CI.

Both deployment workflows create or update resources idempotently, wait for
long-running operations, and test the deployment. Batch invocation waits for a
terminal job state and fails the workflow with parent and child-job diagnostics.
The deployment workflows require an MLflow model and fail with an actionable
error for any other registered model type. For online serving, choose either
the supported Azure ML no-code mode or the default explicit registered
environment. The default prevents Azure ML from creating an anonymous
environment or starting a workspace image build and is required when workspace
storage enforces `allowSharedKeyAccess=false`. The no-code mode does not run an
in-cluster image builder or require root, privileged pods, Linux capabilities,
a Docker socket, or mutable runtime installation by the workflow.
Batch deployment rejects mutable labels, `latest`, unversioned references,
images, and inline Conda environments before any Azure ML operation.

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
      mlflow_no_code: true
      instance_type: ${{ vars.ONLINE_INSTANCE_TYPE }}
      instance_count: 1
      request_file: samples/online-request.json
      runner: ${{ vars.PRIVATE_ARC_RUNNER }}
      tls_ca_key_vault_secret_id: ${{ vars.ONLINE_TLS_CA_SECRET_ID }}
      endpoint_uami_resource_id: ${{ vars.ONLINE_ENDPOINT_UAMI_RESOURCE_ID }}
    secrets:
      AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
      AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
      AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
```

The coordinated infrastructure configuration keys for no-code mode are
`online_compute`, `online_instance_type`, and `online_tls_ca_secret_id`, with
`online_endpoint_uami_resource_id` optional. Default image-only mode additionally
uses `online_environment_name` and `online_environment_version`. Map them to the
uppercase repository/environment variables shown above. Remove any
managed-online VM SKU such as `Standard_DS2_v2`; Kubernetes `instance_type` is a
cluster-defined resource profile. Keep the private ARC runner and existing OIDC
secrets unchanged.

Promote to `test` or `prod` by calling the same reusable workflow with a
different GitHub Environment and environment-scoped OIDC secrets and variables.
