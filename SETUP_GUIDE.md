# PakPolitician-Classifier: Milestone 1 Setup Guide

## Project Status: ✅ READY FOR EXECUTION

This project implements a **Pakistani politician face classification system** using a structured data pipeline with DVC (Data Version Control) and MLflow for experiment tracking.

---

## 🎯 Milestone 1: Data Collection & Preparation

### Dataset Configuration: 16 Politicians

The project is configured for **16 political figures** with the following setup:

| # | Politician | Class Name | Min Images | Target Download |
|---|---|---|---|---|
| 1 | Imran Khan | imran_khan | 80 | 140 |
| 2 | Nawaz Sharif | nawaz_sharif | 80 | 140 |
| 3 | Shehbaz Sharif | shehbaz_sharif | 80 | 140 |
| 4 | Maryam Nawaz | maryam_nawaz | 80 | 140 |
| 5 | Bilawal Bhutto Zardari | bilawal_bhutto_zardari | 80 | 140 |
| 6 | Asif Ali Zardari | asif_ali_zardari | 80 | 140 |
| 7 | Maulana Fazlur Rehman | maulana_fazlur_rehman | 80 | 140 |
| 8 | Altaf Hussain | altaf_hussain | 80 | 140 |
| 9 | Siraj ul Haq | siraj_ul_haq | 80 | 140 |
| 10 | Pervaiz Elahi | pervaiz_elahi | 80 | 140 |
| 11 | Saad Rizvi | saad_rizvi | 80 | 140 |
| 12 | Khawaja Asif | khawaja_asif | 80 | 140 |
| 13 | Gohar Ali Khan | gohar_ali_khan | 80 | 140 |
| 14 | Shahid Khaqan Abbasi | shahid_khaqan_abbasi | 80 | 140 |
| 15 | Asad Umar | asad_umar | 80 | 140 |
| 16 | Ahmed Sharif Chaudhry (DG ISPR) | ahmed_sharif_chaudhry | 80 | 140 |

---

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Git
- 2GB free disk space (for images)

### Dependencies Installation

```powershell
# Navigate to project root
cd C:\Users\Sardar Sameer\PakPolitician-Classifier

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

```

---

## 🚀 Execution Steps

### Option 1: Full Pipeline (Recommended)

Run the entire data collection → split → verification pipeline:

```powershell
# Run from project root
dvc repro
```

This automatically executes:
1. `collect_images` - Download ~140 images per politician
2. `split_dataset` - Split into train (75%), val (15%), test (10%)
3. `verify_dataset` - Verify minimum 80 images per class with proper splits

---

### Option 2: Run Individual Stages

```powershell
# Step 1: Collect images (takes 15-30 minutes depending on internet)
python src/collect_dataset.py

# Step 2: Split collected images into train/val/test
python src/split_dataset.py

# Step 3: Verify dataset integrity
python src/verify_dataset.py
```

---

### Option 3: Alternative Entry Point

```powershell
# This calls collect_dataset.py internally
python src/download_data.py
```

---

## 📊 Expected Output Structure

After successful pipeline execution:

```
PakPolitician-Classifier/
├── data/
│   └── raw/                           # Raw downloaded images
│       ├── imran_khan/
│       ├── nawaz_sharif/
│       ├── ... (16 directories)
│       └── ahmed_sharif_chaudhry/
│
├── dataset/                           # Split dataset
│   ├── train/                        # 75% of images per class
│   │   ├── imran_khan/
│   │   ├── nawaz_sharif/
│   │   └── ... (16 subdirs)
│   ├── val/                          # 15% of images per class
│   │   └── ... (16 subdirs)
│   └── test/                         # 10% of images per class
│       └── ... (16 subdirs)
│
└── reports/
    ├── download_summary.json         # Images downloaded per class
    ├── split_summary.json            # Split statistics per class
    └── dataset_report.json           # Final verification report
```

---

## 📝 Configuration Files

### `params.yaml` - Main Configuration

```yaml
dataset:
  min_images_per_class: 80           # Minimum images required per politician
  download_images_per_class: 140     # Target download (extras for filtering)
  random_state: 42                   # Reproducibility seed
  min_width: 160                     # Min image width in pixels
  min_height: 160                    # Min image height in pixels
  
  split:
    train: 0.75                      # 75% for training
    val: 0.15                        # 15% for validation
    test: 0.10                       # 10% for testing
  
  classes:                           # List of 16 politicians
    - imran_khan
    - nawaz_sharif
    # ... (16 total)
  
  class_queries:                     # Search queries for image download
    imran_khan: "Imran Khan face portrait"
    # ... (customized for each politician)
```

### `dvc.yaml` - Pipeline Definition

Defines three stages:
- **collect_images**: Download politician images
- **split_dataset**: Partition into train/val/test
- **verify_dataset**: Quality assurance checks

---

## ✅ Verification Checklist

After pipeline completes, verify:

- [ ] All 16 politician directories created under `data/raw/`
- [ ] Each class has ≥ 80 images (total)
- [ ] `dataset/train/` has all 16 subdirectories
- [ ] `dataset/val/` has all 16 subdirectories
- [ ] `dataset/test/` has all 16 subdirectories
- [ ] `reports/dataset_report.json` shows all classes valid
- [ ] Image sizes meet minimum (160x160 pixels)

Check final report:
```powershell
Get-Content reports/dataset_report.json | ConvertFrom-Json
```

---

## 🐛 Troubleshooting

### Issue: "Minimum image requirement not met"

**Solution**: 
- Check internet connection
- Increase `download_images_per_class` in params.yaml
- Run `python src/collect_dataset.py` again (resumes with existing images)
- Check `reports/download_summary.json` for download status

### Issue: DVC command not recognized

**Solution**:
```powershell
# Make sure DVC is installed
pip install dvc>=3.50

# Initialize DVC in project
dvc init

# View DVC configuration
dvc config --list
```

### Issue: Images too small or corrupt

**Solution**: Already handled by the pipeline:
- Minimum size filter: 160x160 pixels
- Corrupted images automatically skipped
- Duplicate images detected by SHA1 hash

---

## 📈 Next Steps (Milestone 2)

After Milestone 1 completes:

1. **Feature Extraction**: Convert face images to feature vectors
   - Using pre-trained face recognition models (e.g., FaceNet, VGGFace2)
   - Save as CSV for model training

2. **Model Training**: Train classifier on face features
   - Logistic Regression / Random Forest / Neural Networks
   - Track experiments with MLflow

3. **Model Evaluation**: Test on unseen data
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix analysis

---

## 📞 Support

For issues or questions:
1. Check `reports/dataset_report.json` for data validation errors
2. Verify params.yaml configuration
3. Ensure internet connectivity for image download
4. Check disk space availability

---

## 🎓 Project Architecture

```
DVC Pipeline: collect_images → split_dataset → verify_dataset
   ↓
   ├─ Input: params.yaml (16 politicians, 140 images each)
   ├─ Storage: data/raw/, dataset/train,val,test/
   └─ Output: reports/dataset_report.json (validation metrics)
```

---

**Status**: ✅ Ready for execution  
**Created**: 2026-05-09  
**Milestone**: 1 (Data Collection & Preparation)  
**Target Classes**: 16 Pakistani political figures
