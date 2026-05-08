from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add project root so root-level modules are importable when running from src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mlflow_setup import setup_mlflow


def main() -> None:
    test_df = pd.read_csv(Path("data") / "processed" / "test.csv")
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    model = joblib.load(Path("models") / "model.joblib")
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    setup_mlflow()
    with mlflow.start_run(run_name="evaluate-logreg"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_dir / "metrics.json"))

    print("Evaluation complete. Metrics saved to metrics/metrics.json")


if __name__ == "__main__":
    main()
