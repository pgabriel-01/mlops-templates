# Azure DevOps Azure ML CLI v2 templates

These templates support the Classical AML CLI v2 training, online endpoint, and batch endpoint pipelines.

All operation templates accept explicit Azure DevOps and workspace identity parameters:

- `azure_service_connection`
- `resource_group`
- `workspace_name`

Endpoint operations additionally accept:

- `endpoint_type`: `online` or `batch`
- `endpoint_name`

Defaults preserve the legacy pipeline variables, but new consumers should pass explicit values.

## Setup

`setup.yml` verifies Azure CLI, installs or upgrades the Azure ML CLI extension, verifies the target workspace, and configures CLI defaults.

```yaml
- template: templates/aml-cli-v2/setup.yml@mlops-templates
  parameters:
    azure_service_connection: Azure-ARM-Dev
    resource_group: $(resource_group)
    workspace_name: $(aml_workspace)
```

## Training order

1. `setup.yml`
2. `register-environment.yml`
3. `register-data.yml`
4. `create-compute.yml` only when Terraform does not own the named compute
5. `run-pipeline.yml`

`run-pipeline.yml` names its task `trainingJob` and exposes `job_id` and `job_status` output variables. A non-completed terminal state fails the task.

## Online endpoint order

1. `setup.yml`
2. `create-endpoint.yml`
3. `create-deployment.yml`
4. `allocate-traffic.yml`
5. `test-deployment.yml`

## Batch endpoint order

1. `setup.yml`
2. `create-compute.yml` when the batch deployment requires separate compute
3. `create-endpoint.yml`
4. `create-deployment.yml`
5. `test-deployment.yml`

Project-specific environment, data, pipeline, deployment, endpoint, request, and scoring files remain in the consuming project repository.
