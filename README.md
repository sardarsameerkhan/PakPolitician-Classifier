# PakPolitician-Classifier

Starter machine learning pipeline scaffold for classifying Pakistani politicians.

## What Is Included

- DVC pipeline with 3 stages: `prepare`, `train`, `evaluate`
- MLflow experiment setup helper
- Parameterized training via `params.yaml`
- Metrics output for DVC tracking

## Project Structure

```
.
|-- .gitignore
|-- dvc.yaml
|-- mlflow_setup.py
|-- params.yaml
|-- requirements.txt
`-- src/
	|-- prepare_data.py
	|-- train.py
	`-- evaluate.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Initialize DVC if not already done:

```powershell
dvc init
```

## Run Pipeline

```powershell
dvc repro
```

This will generate:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/model.joblib`
- `metrics/metrics.json`

## Run Individual Steps

```powershell
python src/prepare_data.py
python src/train.py
python src/evaluate.py
```

## MLflow

By default, MLflow uses local tracking in `mlruns/` and experiment name `pakpolitician-classifier`.

