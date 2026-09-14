# Azure DevOps Terraform templates

Generated platform bootstrap pipelines may default to `main` while exposing the ref as a compile-time parameter:

```yaml
parameters:
  - name: templatesRef
    type: string
    default: refs/heads/main

resources:
  repositories:
    - repository: mlops-templates
      type: git
      name: mlops-templates
      ref: ${{ parameters.templatesRef }}
```

Release pipelines should use an immutable repository ref:

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

`terraform-state-bootstrap.yml` creates or verifies the Azure Storage backend and creates the blob container through the Azure Resource Manager management plane. It makes no Blob data-plane call during bootstrap.

Required parameters:

| Parameter | Purpose |
| --- | --- |
| `azureServiceConnection` | Azure Resource Manager workload identity federation service connection |
| `location` | Backend resource location |
| `backendResourceGroup` | Terraform state resource group |
| `backendStorageAccount` | Terraform state storage account |
| `backendContainer` | Terraform state blob container |
| `allowPublicNetworkAccess` | Enables or disables the storage public endpoint; defaults to `true` |
| `cicdPrincipalObjectId` | Optional Entra service-principal object ID granted Storage Blob Data Contributor |
| `roleAssignmentPropagationDelaySeconds` | Delay after creating a new data-plane role assignment; defaults to `60` |

The connection needs Azure Resource Manager permission to create the resource group, storage account, `blobServices/containers` child resource, and optional role assignment. Shared-key access is disabled, HTTPS is required, and the minimum TLS version is 1.2. Container creation uses the ARM management plane; bootstrap does not call the Blob data plane.

## Managed DevOps platform bootstrap

`bicep/managed-devops-platform.bicep` deploys a dedicated Managed DevOps Pool and keyless Terraform state backend. `managed-devops-platform.yml` is its direct Azure DevOps wrapper. `platform-bootstrap.yml` selects DEV, Test, or Prod service-connection and CI/CD principal inputs without hardcoding live IDs.

Private mode, the default, creates:

- a VNet with a subnet delegated to `Microsoft.DevOpsInfrastructure/pools`
- a separate private-endpoint subnet
- the `privatelink.blob.core.windows.net` private DNS zone and VNet link
- a Blob private endpoint and DNS zone group
- state storage with shared keys disabled, OAuth-compatible authorization, HTTPS-only, TLS 1.2, and public access disabled
- Reader and Network Contributor assignments for the `DevOpsInfrastructure` service principal on the VNet
- Storage Blob Data Contributor for the selected environment CI/CD principal

Public mode is explicit with `networkMode: public`. It omits the VNet/private endpoint/DNS resources and enables the storage public endpoint while retaining keyless authentication.

The bootstrap pipeline runs on an existing Microsoft-hosted or bootstrap agent. After it succeeds, later stages or pipelines use the emitted `agent_pool_name`, `terraform_st_resource_group`, `terraform_st_storage_account`, and `terraform_st_container_name` values. Cross-job references use the task name `managedDevOpsPlatform`.

The wrapper checks out the `mlops-templates` repository resource to `s/mlops-templates` so the Bicep asset is available. Override `templateRepository`, `templateCheckoutPath`, and `templateFile` together when the repository alias or checkout layout differs.

```yaml
- template: templates/infra/platform-bootstrap.yml@mlops-templates
  parameters:
    environment: dev
    devAzureServiceConnection: Azure-ARM-Dev
    testAzureServiceConnection: Azure-ARM-Test
    prodAzureServiceConnection: Azure-ARM-Prod
    devCicdPrincipalObjectId: $(dev_cicd_principal_object_id)
    testCicdPrincipalObjectId: $(test_cicd_principal_object_id)
    prodCicdPrincipalObjectId: $(prod_cicd_principal_object_id)
    devOpsInfrastructurePrincipalObjectId: $(devops_infrastructure_principal_object_id)
    location: eastus2
    resourceGroup: rg-mlops-platform-dev
    networkMode: private
    virtualNetworkName: vnet-mlops-platform-dev
    stateStorageAccountName: stmlopsplatformdev
    managedDevOpsPoolName: mdp-mlops-dev
    managedDevOpsPoolAlias: mlops-private-dev
    devCenterProjectResourceId: $(dev_center_project_resource_id)
```

The VNet and pool must use the same region. The delegated subnet is exclusive to one Managed DevOps Pool and must not use `172.17.0.0/16`, which the service reserves for internal operations.

`azureDevOpsOrganizationUrl` and `azureDevOpsProjectName` default to `$(System.CollectionUri)` and `$(System.TeamProjectId)` respectively, so pipelines do not commit organization or project identifiers.

Private deployment ordering:

1. Run `platform-bootstrap.yml` on an existing Microsoft-hosted or bootstrap pool.
2. Wait for the Managed DevOps Pool to be available in the selected Azure DevOps project.
3. Queue Terraform and AML stages with `pool.name` set to the configured pool alias.
4. Use `terraform-deploy.yml`; its Azure AD backend access resolves through the Blob private endpoint and private DNS zone.

`devCenterProjectResourceId` refers to an existing Dev Center project in the same region. The bootstrap identity must be able to register `Microsoft.DevOpsInfrastructure`, deploy resources, and create role assignments.

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

`terraformVersion` accepts `latest`, an exact semantic version such as `1.16.2`, or a stable-patch range such as `1.16.x`. The range form resolves at runtime from HashiCorp's official Terraform release index and sets `terraformResolvedVersion` before `TerraformInstaller@1` runs.

```yaml
- template: templates/infra/terraform-state-bootstrap.yml@mlops-templates
  parameters:
    azureServiceConnection: Azure-ARM-Dev
    location: eastus2
    backendResourceGroup: rg-taxi-dev-tf
    backendStorageAccount: sttaxidevtf
    backendContainer: default
    cicdPrincipalObjectId: $(cicd_principal_object_id)
    allowPublicNetworkAccess: true

- template: templates/infra/terraform-deploy.yml@mlops-templates
  parameters:
    azureServiceConnection: Azure-ARM-Dev
    terraformVersion: 1.16.x
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
