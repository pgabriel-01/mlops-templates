targetScope = 'resourceGroup'

@allowed([
  'public'
  'private'
])
param networkMode string = 'private'

param location string = resourceGroup().location
param virtualNetworkName string
param virtualNetworkAddressPrefix string = '10.20.0.0/16'
param managedPoolSubnetName string = 'snet-managed-devops'
param managedPoolSubnetPrefix string = '10.20.1.0/24'
param privateEndpointSubnetName string = 'snet-private-endpoints'
param privateEndpointSubnetPrefix string = '10.20.2.0/24'
param privateDnsZoneName string = 'privatelink.blob.${environment().suffixes.storage}'
param stateStorageAccountName string
param stateContainerName string = 'default'
param managedDevOpsPoolName string
param managedDevOpsPoolAlias string = managedDevOpsPoolName
param managedDevOpsPoolSku string = 'Standard_D2ads_v5'
param managedDevOpsPoolImage string = 'ubuntu-24.04'
@minValue(1)
@maxValue(10000)
param maximumConcurrency int = 2
param devCenterProjectResourceId string
param azureDevOpsOrganizationUrl string
param azureDevOpsProjectName string
param devOpsInfrastructurePrincipalObjectId string
param cicdPrincipalObjectId string

var privateMode = networkMode == 'private'
var readerRoleDefinitionId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'acdd72a7-3385-48ef-bd42-f606fba81ae7'
)
var networkContributorRoleDefinitionId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4d97b98b-1d4f-4787-a291-c67834d212e7'
)
var storageBlobDataContributorRoleDefinitionId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
)

resource stateStorageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: stateStorageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: privateMode ? 'Disabled' : 'Enabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: privateMode ? 'Deny' : 'Allow'
    }
  }
}

resource stateBlobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  name: 'default'
  parent: stateStorageAccount
}

resource stateContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: stateContainerName
  parent: stateBlobService
  properties: {
    publicAccess: 'None'
  }
}

resource cicdStateRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(stateStorageAccount.id, cicdPrincipalObjectId, storageBlobDataContributorRoleDefinitionId)
  scope: stateStorageAccount
  properties: {
    principalId: cicdPrincipalObjectId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataContributorRoleDefinitionId
  }
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-05-01' = if (privateMode) {
  name: virtualNetworkName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        virtualNetworkAddressPrefix
      ]
    }
  }
}

resource managedPoolSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' = if (privateMode) {
  name: managedPoolSubnetName
  parent: virtualNetwork
  properties: {
    addressPrefix: managedPoolSubnetPrefix
    delegations: [
      {
        name: 'managed-devops-pools'
        properties: {
          serviceName: 'Microsoft.DevOpsInfrastructure/pools'
        }
      }
    ]
  }
}

resource privateEndpointSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' = if (privateMode) {
  name: privateEndpointSubnetName
  parent: virtualNetwork
  properties: {
    addressPrefix: privateEndpointSubnetPrefix
    privateEndpointNetworkPolicies: 'Disabled'
  }
}

resource devOpsInfrastructureReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (privateMode) {
  name: guid(virtualNetwork.id, devOpsInfrastructurePrincipalObjectId, readerRoleDefinitionId)
  scope: virtualNetwork
  properties: {
    principalId: devOpsInfrastructurePrincipalObjectId
    principalType: 'ServicePrincipal'
    roleDefinitionId: readerRoleDefinitionId
  }
}

resource devOpsInfrastructureNetworkContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (privateMode) {
  name: guid(virtualNetwork.id, devOpsInfrastructurePrincipalObjectId, networkContributorRoleDefinitionId)
  scope: virtualNetwork
  properties: {
    principalId: devOpsInfrastructurePrincipalObjectId
    principalType: 'ServicePrincipal'
    roleDefinitionId: networkContributorRoleDefinitionId
  }
}

resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = if (privateMode) {
  name: privateDnsZoneName
  location: 'global'
}

resource blobPrivateDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = if (privateMode) {
  name: '${virtualNetworkName}-link'
  parent: blobPrivateDnsZone
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource blobPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (privateMode) {
  name: 'pe-${stateStorageAccountName}-blob'
  location: location
  properties: {
    subnet: {
      id: privateEndpointSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'blob'
        properties: {
          groupIds: [
            'blob'
          ]
          privateLinkServiceId: stateStorageAccount.id
        }
      }
    ]
  }
}

resource blobPrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (privateMode) {
  name: 'default'
  parent: blobPrivateEndpoint
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'blob'
        properties: {
          privateDnsZoneId: blobPrivateDnsZone.id
        }
      }
    ]
  }
}

resource managedDevOpsPool 'Microsoft.DevOpsInfrastructure/pools@2025-09-20' = {
  name: managedDevOpsPoolName
  location: location
  properties: {
    maximumConcurrency: maximumConcurrency
    devCenterProjectResourceId: devCenterProjectResourceId
    agentProfile: {
      kind: 'Stateless'
    }
    organizationProfile: {
      kind: 'AzureDevOps'
      alias: managedDevOpsPoolAlias
      organizations: [
        {
          url: azureDevOpsOrganizationUrl
          projects: [
            azureDevOpsProjectName
          ]
          parallelism: maximumConcurrency
          openAccess: false
        }
      ]
      permissionProfile: {
        kind: 'CreatorOnly'
        users: []
        groups: []
      }
    }
    fabricProfile: {
      kind: 'Vmss'
      sku: {
        name: managedDevOpsPoolSku
      }
      images: [
        {
          wellKnownImageName: managedDevOpsPoolImage
        }
      ]
      osProfile: {
        logonType: 'Service'
      }
      storageProfile: {
        osDiskStorageAccountType: 'StandardSSD'
      }
      networkProfile: privateMode
        ? {
            subnetId: managedPoolSubnet.id
          }
        : null
    }
  }
  dependsOn: privateMode
    ? [
        devOpsInfrastructureReader
        devOpsInfrastructureNetworkContributor
        blobPrivateDnsZoneGroup
      ]
    : []
}

output managedDevOpsPoolName string = managedDevOpsPool.name
output managedDevOpsPoolAlias string = managedDevOpsPoolAlias
output stateStorageAccountName string = stateStorageAccount.name
output stateContainerName string = stateContainer.name
output virtualNetworkId string = privateMode ? virtualNetwork.id : ''
output managedPoolSubnetId string = privateMode ? managedPoolSubnet.id : ''
output privateEndpointSubnetId string = privateMode ? privateEndpointSubnet.id : ''
