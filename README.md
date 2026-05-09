# PakPolitician-Classifier

Milestone 1 in this repository focuses on collecting and preparing an image dataset for Pakistani politician face classification.

## Scope Of This Milestone

- Collect image data for 16 classes (15 politicians + 1 military spokesperson)
- Enforce minimum 80 images per class
- Split into `train/val/test` with `75/15/10`
- Track dataset stages with DVC

## Dataset Layout

```
dataset/
|-- train/
|   |-- imran_khan/
|   |-- nawaz_sharif/
|   `-- ...
|-- val/
|   |-- imran_khan/
|   `-- ...
`-- test/
    |-- imran_khan/
    `-- ...
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Initialize DVC:

```powershell
dvc init
```

## Data Collection Pipeline

Run all stages:

```powershell
dvc repro
```

Or run one-by-one:

```powershell
python src/collect_dataset.py
python src/split_dataset.py
python src/verify_dataset.py
```

Generated outputs:

- `data/raw/<class_name>/...`
- `dataset/train/<class_name>/...`
- `dataset/val/<class_name>/...`
- `dataset/test/<class_name>/...`
- `reports/download_summary.json`
- `reports/split_summary.json`
- `reports/dataset_report.json`

## Notes

- Edit class names and search queries in `params.yaml`.
- Keep only publicly available, lawful images from allowed sources.
- Perform augmentation only on the training split during model training.

