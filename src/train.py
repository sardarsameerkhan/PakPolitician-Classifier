from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add project root so root-level modules are importable when running from src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mlflow_setup import setup_mlflow


def load_params() -> dict:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    random_state = int(params["data"]["random_state"])
    max_iter = int(params["train"]["max_iter"])
    c_value = float(params["train"]["C"])

    train_df = pd.read_csv(Path("data") / "processed" / "train.csv")
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    setup_mlflow()
    with mlflow.start_run(run_name="train-logreg"):
        model = LogisticRegression(max_iter=max_iter, C=c_value, random_state=random_state)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)

        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        mlflow.log_params(
            {
                "random_state": random_state,
                "max_iter": max_iter,
                "C": c_value,
            }
        )
        mlflow.log_metric("train_accuracy", float(train_accuracy))
        mlflow.log_artifact(str(model_path))

        train_metrics = {"train_accuracy": float(train_accuracy)}
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with (metrics_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(train_metrics, f, indent=2)

    print("Training complete. Model saved to models/model.joblib")


if __name__ == "__main__":
    main()
