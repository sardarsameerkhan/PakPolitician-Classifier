# Model Training Guide

This guide explains how to train and evaluate the politician face classifier models.

## Prerequisites

- Python 3.10+
- Virtual environment activated (`.venv\Scripts\activate` on Windows)
- All dependencies installed: `pip install -r requirements.txt`

## Dataset Structure

```
dataset/
├── train/          # 1,063 images (75% of dataset)
│   ├── politician_1/
│   ├── politician_2/
│   └── ...
├── val/            # 206 images (15% of dataset)
│   └── ...
└── test/           # 152 images (10% of dataset)
    └── ...
```

## Pipeline Components

### 1. Data Loading & Augmentation

**File:** `src/augmentation.py` and `src/dataset.py`

**Features:**
- 7-technique augmentation for training data:
  - Random resized crop (zoom 0.8-1.0)
  - Rotation (±30°)
  - Horizontal/vertical flips
  - Brightness/contrast variation
  - Gaussian noise
  - Blur
  - ImageNet normalization
- Validation/test data: Resize + Normalization only (no augmentation)

**Test:**
```bash
python verify_pipeline.py
```

Expected output:
- 1,063 training images across 16 classes
- 206 validation images
- 152 test images
- Batch shape: [32, 3, 224, 224]

### 2. Model Training

**File:** `src/train_models.py`

**Models Implemented:**
1. **ResNet-50** - 50-layer residual network (pretrained ImageNet)
2. **EfficientNet-B0** - Efficient mobile network (pretrained ImageNet)

**Training Configuration:**
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau (factor=0.1, patience=3)
- Batch size: 32
- Image size: 224×224

**Quick Test (2 epochs):**
```bash
python quick_train_test.py
```

Outputs:
- `models_test/ResNet-50_best.pth` - Best model checkpoint

**Full Training (50 epochs):**
```bash
cd src
python train_models.py
```

Expected outputs:
- `models/ResNet-50_best.pth`
- `models/EfficientNet-B0_best.pth`
- `reports/training_results.json`

### 3. Model Evaluation

**File:** `src/evaluate_models.py`

**Metrics Generated:**
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix heatmap
- Top 5 misclassified samples with images
- Training vs validation curves (loss and accuracy)

**Running Evaluation:**
```bash
cd src
python evaluate_models.py
```

Expected outputs:
- `reports/ResNet-50_evaluation.json`
- `reports/ResNet-50_report.txt`
- `reports/ResNet-50_confusion_matrix.png`
- `reports/ResNet-50_misclassified.png`
- `reports/ResNet-50_training_curves.png`
- `reports/EfficientNet-B0_evaluation.json`
- `reports/EfficientNet-B0_report.txt`
- `reports/EfficientNet-B0_confusion_matrix.png`
- `reports/EfficientNet-B0_misclassified.png`
- `reports/EfficientNet-B0_training_curves.png`

## Performance Targets

- **Accuracy Goal:** ≥90% on test set
- **Per-class F1:** ≥0.85 for each politician
- **Model Comparison:** Best model selection based on validation accuracy

## MLflow Tracking

Training metrics are automatically logged to MLflow:
- Model name
- Epochs, batch size, number of classes
- Per-epoch train/val loss and accuracy
- Best model artifacts

View MLflow UI:
```bash
mlflow ui
```

## Complete Training Pipeline

```bash
# 1. Verify dataset
python verify_pipeline.py

# 2. Train models (ResNet-50 and EfficientNet-B0)
cd src
python train_models.py

# 3. Evaluate models
python evaluate_models.py

# 4. View results
cat ../reports/ResNet-50_report.txt
cat ../reports/EfficientNet-B0_report.txt
```

## Troubleshooting

### CUDA/GPU Issues
- If using CPU: Training is slower but will work
- To force CPU: `export CUDA_VISIBLE_DEVICES=-1` (Linux/Mac) or `set CUDA_VISIBLE_DEVICES=-1` (Windows)

### Out of Memory
- Reduce batch_size in `get_dataloaders()` call
- Reduce num_workers to 0

### Missing Classes
- Ensure `dataset/train/`, `dataset/val/`, `dataset/test/` folders exist
- Ensure each has 16 politician subfolders with images

## File Locations

- **Training logs:** `mlruns/` (MLflow)
- **Best models:** `models/`
- **Reports:** `reports/`
- **Test models:** `models_test/` (from quick test)

## Next Steps

1. Analyze evaluation metrics in `reports/` folder
2. Compare model performance (ResNet-50 vs EfficientNet-B0)
3. If needed, retrain with adjusted hyperparameters
4. Deploy best-performing model
