# Azure MLOps (v2) Solution Accelerator

Welcome to the MLOps (v2) solution accelerator repository! This project is intended to serve as the starting point for MLOps implementation in Azure. The main repository from where you can start can be found in this [main README file](https://github.com/Azure/mlops-v2/blob/main/README.md)

Reusable Python SDK v2 batch deployments require consumer-owned scoring code
and an immutable Azure ML environment. See the
[batch deployment contract](docs/python-sdk-v2-batch.md) for workflow inputs,
path validation, and the required MLflow scoring signature.

Reusable Python SDK v2 Kubernetes online deployments support an explicit
`mlflow_no_code` mode. Azure ML supplies the curated inference environment for
the registered MLflow model, so the workflow omits both custom scoring code and
the deployment environment instead of attempting an unsupported in-cluster
runtime image build. The existing immutable digest-pinned image-only mode
remains the default. See the
[Python SDK v2 workflow documentation](templates/python-sdk-v2/README.md).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow.
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
