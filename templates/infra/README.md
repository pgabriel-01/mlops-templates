# Azure DevOps Terraform templates

Use an immutable repository ref when consuming these templates:

```yaml
resources:
  repositories:
    - repository: mlops-templates
      type: git
      name: mlops-templates
      ref: refs/tags/v1.0.0
```

During integration, pin an exact commit SHA until a release tag is available.

## State bootstrap

`terraform-state-bootstrap.yml` creates or verifies the Azure Storage backend and grants the workload identity `Storage Blob Data Contributor`.

Required parameters:

| Parameter | Purpose |
| --- | --- |
| `azureServiceConnection` | Azure Resource Manager workload identity federation service connection |
| `location` | Backend resource location |
| `backendResourceGroup` | Terraform state resource group |
| `backendStorageAccount` | Terraform state storage account |
| `backendContainer` | Terraform state blob container |
| `cicdPrincipalObjectId` | Entra service-principal object ID used for role assignment |

The connection must be able to create the backend resources and role assignment. Role assignment creation requires Owner or User Access Administrator.
`cicdPrincipalObjectId` is required because AzureCLI's `servicePrincipalId` is the application/client ID, while `--assignee-object-id` requires the Entra service-principal object ID.

## Composed deployment

`terraform-deploy.yml` installs Terraform, initializes the OIDC/Azure AD backend, runs format and validation checks, saves a plan, and optionally applies that exact plan.

The project Terraform root must accept:

- `location`
- `prefix`
- `postfix`
- `environment`
- `project_number`
- `cicd_principal_object_id`
- `enable_aml_computecluster`
- `aml_compute_sku`
- `enable_monitoring`
- `enable_private_endpoints`
- `vnet_address_space`
- `training_subnet_address_prefix`
- `endpoints_subnet_address_prefix`

When `apply: true`, the root must expose:

- `resource_group_name`
- `aml_workspace_name`
- `storage_account_name`
- `training_compute_name`

The apply task is named `terraformOutputs`; its output variables use the same names. It also sets the same-job compatibility variables `resource_group`, `aml_workspace`, `storage_account`, and `training_compute`.

```yaml
- template: templates/infra/terraform-state-bootstrap.yml@mlops-templates
  parameters:
    azureServiceConnection: Azure-ARM-Dev
    location: eastus2
    backendResourceGroup: rg-taxi-dev-tf
    backendStorageAccount: sttaxidevtf
    backendContainer: default
    cicdPrincipalObjectId: $(cicd_principal_object_id)

- template: templates/infra/terraform-deploy.yml@mlops-templates
  parameters:
    azureServiceConnection: Azure-ARM-Dev
    terraformVersion: 1.14.3
    workingDirectory: $(System.DefaultWorkingDirectory)/infrastructure/terraform
    backendResourceGroup: rg-taxi-dev-tf
    backendStorageAccount: sttaxidevtf
    backendContainer: default
    backendKey: classical-dev.tfstate
    location: eastus2
    prefix: taxi
    postfix: '10001'
    environment: dev
    projectNumber: '001'
    cicdPrincipalObjectId: $(cicd_principal_object_id)
    enableAmlComputeCluster: true
    amlComputeSku: STANDARD_D2S_V3
    enableMonitoring: true
    enablePrivateEndpoints: false
    vnetAddressSpace: 10.0.0.0/16
    trainingSubnetAddressPrefix: 10.0.1.0/24
    endpointsSubnetAddressPrefix: 10.0.2.0/24
```

`cicdPrincipalObjectId` must be the Entra service-principal object ID, not the application/client ID exposed as `servicePrincipalId` by `AzureCLI@2`.

The lower-level templates remain available for compatibility. `run-terraform-plan.yml` accepts the same `projectNumber` and `cicdPrincipalObjectId` values when `includeProjectMetadata: true`; the flag defaults to `false` so Terraform roots that do not declare those variables continue to work. New consumers should use the composed templates.
