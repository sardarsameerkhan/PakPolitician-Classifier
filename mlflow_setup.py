from __future__ import annotations

from pathlib import Path

import mlflow


def default_tracking_uri() -> str:
	"""Return a local file-based MLflow tracking URI."""
	tracking_dir = Path("mlruns").resolve()
	return tracking_dir.as_uri()


def setup_mlflow(experiment_name: str = "pakpolitician-classifier") -> str:
	"""Configure MLflow tracking and experiment for local development."""
	tracking_uri = default_tracking_uri()
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)
	return tracking_uri


if __name__ == "__main__":
	uri = setup_mlflow()
	print(f"MLflow configured: {uri}")
