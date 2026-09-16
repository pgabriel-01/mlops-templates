# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pandas as pd

MODEL: Any = None


def _find_mlflow_model_path(model_root: Path) -> Path:
    if (model_root / "MLmodel").is_file():
        return model_root

    candidates = sorted(path.parent for path in model_root.rglob("MLmodel"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one MLflow MLmodel file under AZUREML_MODEL_DIR; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def _load_mlflow_model(model_path: Path) -> Any:
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(str(model_path))


def init() -> None:
    global MODEL

    model_root = Path(os.environ["AZUREML_MODEL_DIR"])
    MODEL = _load_mlflow_model(_find_mlflow_model_path(model_root))


def _read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(
        f"Unsupported batch input format '{suffix}' for {path}. "
        "Use CSV, Parquet, JSON, or JSON Lines."
    )


def _prediction_frame(predictions: Any) -> pd.DataFrame:
    if isinstance(predictions, pd.DataFrame):
        frame = predictions.reset_index(drop=True).copy()
    elif isinstance(predictions, pd.Series):
        frame = predictions.reset_index(drop=True).to_frame(name="prediction")
    else:
        try:
            frame = pd.DataFrame(predictions)
        except ValueError:
            frame = pd.DataFrame([predictions])

    if all(isinstance(column, int) for column in frame.columns):
        frame.columns = [
            "prediction" if len(frame.columns) == 1 else f"prediction_{index}"
            for index in range(len(frame.columns))
        ]
    return frame


def run(mini_batch: list[str]) -> pd.DataFrame:
    if MODEL is None:
        raise RuntimeError("Batch scoring model is not initialized.")

    outputs = []
    for input_file in mini_batch:
        path = Path(input_file)
        predictions = _prediction_frame(MODEL.predict(_read_input(path)))
        predictions.insert(0, "source_row", range(len(predictions)))
        predictions.insert(0, "source_file", path.name)
        outputs.append(predictions)

    if not outputs:
        return pd.DataFrame(columns=["source_file", "source_row", "prediction"])
    return pd.concat(outputs, ignore_index=True)
