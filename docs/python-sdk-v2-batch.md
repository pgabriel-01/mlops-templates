# Python SDK v2 batch deployment contract

The reusable `.github/workflows/python-sdk-v2-batch.yml` workflow deploys a
registered MLflow model with an explicit immutable Azure ML environment and
consumer-supplied scoring code. This prevents Azure ML from treating the
deployment as an environment-only, no-code deployment and synthesizing an
anonymous image build.

## Required scoring inputs

- `scoring_code_directory`: traversal-free path inside the checked-out consumer
  repository. Use a dedicated directory such as `src/batch_scoring`; the
  repository root and `.mlops-python-sdk` reusable checkout are rejected.
- `scoring_script`: traversal-free file path relative to
  `scoring_code_directory`, such as `score.py`.
- `deployment_environment`: exact numeric-version Azure ML environment
  reference. The default remains
  `azureml://registries/azureml/environments/sklearn-1.5/versions/53`.

For a registry URI, the deployment script creates a registry-scoped `MLClient`
with the same authenticated credential used by the workspace client, fetches
the exact environment name and numeric version, and reads the registry client's
authoritative subscription, resource group, and registry operation scope. The
SDK can return registry shorthand in `Environment.id`, so the script validates
that lookup result, constructs the full ARM ID from the canonical operation
scope, and assigns it to the fetched `Environment` entity. This entity is passed
to `BatchDeployment` because SDK dependency orchestration rejects raw registry
ARM ID strings; the SDK then extracts the corrected entity ID and writes the
full ARM string to `ModelConfiguration.EnvironmentId`. Assignment/readback
failure and missing, unsafe, or mismatched scope data fail closed.
Workspace `azureml:<name>:<version>` references and full workspace or registry
environment resource IDs remain supported without rewriting.

Both scoring paths must exist after the consumer checkout. Absolute paths,
`..` traversal, symlink escapes, and missing files are rejected before an
Azure ML client is created.

A generic copy-ready implementation is available at
`examples/python-sdk-v2/batch-scoring/score.py`. Consumers should copy and
version the scoring directory in their own repository, adapt feature/result
columns for their registered model, and pass that consumer path to the reusable
workflow. The reusable SDK checkout is intentionally not accepted as the
deployment code source.

## Scoring script contract

The scoring module must define:

```python
def init():
    ...


def run(mini_batch):
    ...
```

`init()` must load the mounted registered MLflow model from
`AZUREML_MODEL_DIR`, for example by locating the MLmodel directory beneath that
mount and calling `mlflow.pyfunc.load_model`.

`run(mini_batch)` receives a list of input file paths. It must read every
supported Parquet or CSV file in stable input order, produce one deterministic
prediction row per input row, and either:

1. return a nonempty pandas `DataFrame`, which Azure ML appends to the
   deployment `output_file_name` (`predictions.csv` by default); or
2. use another output behavior explicitly supported and configured by Azure ML
   batch endpoints.

For the `taxi-model` contract, the returned frame should retain a stable row
identifier when available and include a consistently named prediction column.
Empty mini-batches, unreadable inputs, unsupported file types, missing model
mounts, and empty prediction results must raise an actionable exception rather
than returning an empty success.

After Azure ML reports the deployment update as succeeded, the reusable script
reads it back and requires a matching non-null environment plus a non-null code
configuration and matching scoring script before setting the endpoint default
or allowing invocation.
