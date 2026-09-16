import os
from pathlib import Path

import mlflow
import pandas as pd

_model = None


def init():
    global _model

    model_mount = Path(os.environ["AZUREML_MODEL_DIR"]).resolve()
    model_metadata = sorted(model_mount.rglob("MLmodel"))
    if not model_metadata:
        raise RuntimeError(
            f"No MLflow MLmodel file found below AZUREML_MODEL_DIR={model_mount}"
        )
    _model = mlflow.pyfunc.load_model(str(model_metadata[0].parent))


def run(mini_batch):
    if _model is None:
        raise RuntimeError("The MLflow model was not initialized")
    if not mini_batch:
        raise ValueError("Azure ML supplied an empty mini-batch")

    outputs = []
    for input_file in mini_batch:
        path = Path(input_file)
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            features = pd.read_parquet(path)
        elif suffix == ".csv":
            features = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported batch input file type: {path}")
        if features.empty:
            raise ValueError(f"Batch input file contains no rows: {path}")

        predictions = _model.predict(features)
        if isinstance(predictions, pd.DataFrame):
            prediction_frame = predictions.reset_index(drop=True)
        elif isinstance(predictions, pd.Series):
            prediction_frame = predictions.rename("prediction").to_frame()
        else:
            prediction_frame = pd.DataFrame({"prediction": predictions})
        if len(prediction_frame.index) != len(features.index):
            raise RuntimeError(
                f"Model returned {len(prediction_frame.index)} predictions for "
                f"{len(features.index)} input rows from {path}"
            )

        prediction_frame.insert(0, "source_row", range(len(features.index)))
        prediction_frame.insert(0, "source_file", path.name)
        outputs.append(prediction_frame)

    result = pd.concat(outputs, ignore_index=True)
    if result.empty:
        raise RuntimeError("Batch scoring produced no prediction rows")
    return result
