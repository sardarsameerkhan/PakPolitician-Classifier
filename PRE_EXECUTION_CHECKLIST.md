# Pre-Execution Checklist

Use this checklist before running `dvc repro` to ensure everything is ready.

## ✅ System Requirements

- [ ] Python 3.8 or higher installed
  ```powershell
  python --version  # Should show 3.8+
  ```

- [ ] Windows PowerShell (or equivalent terminal)

- [ ] 2-3 GB free disk space available
  ```powershell
  Get-Volume C: | Select-Object SizeRemaining  # Check free space
  ```

- [ ] Stable internet connection (for image downloads)

- [ ] Administrator access (for pip installations)

---

## ✅ Project Setup

- [ ] Navigate to project root
  ```powershell
  cd C:\Users\Sardar Sameer\PakPolitician-Classifier
  ```

- [ ] Virtual environment created
  ```powershell
  python -m venv venv
  ```

- [ ] Virtual environment activated
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

- [ ] Dependencies installed
  ```powershell
  pip install -r requirements.txt
  ```

- [ ] DVC initialized
  ```powershell
  dvc init
  ```

---

## ✅ Configuration Files Verified

- [ ] `params.yaml` exists with 16 politicians defined
  ```powershell
  Test-Path params.yaml  # Should return True
  ```

- [ ] `dvc.yaml` exists with 3 stages defined
  ```powershell
  Test-Path dvc.yaml  # Should return True
  ```

- [ ] `requirements.txt` contains all dependencies
  ```powershell
  Get-Content requirements.txt
  # Should include: duckduckgo-search, pillow, pyyaml, dvc, etc.
  ```

---

## ✅ Python Environment Verified

- [ ] Correct Python is being used
  ```powershell
  python -c "import sys; print(sys.executable)"
  # Should show path to venv/Scripts/python.exe
  ```

- [ ] All required packages installed
  ```powershell
  pip list | findstr /I "duckduckgo pillow dvc mlflow pandas"
  # All should be listed
  ```

- [ ] DuckDuckGo search package working
  ```powershell
  python -c "from duckduckgo_search import DDGS; print('✓ DuckDuckGo working')"
  ```

- [ ] PIL (Pillow) image processing working
  ```powershell
  python -c "from PIL import Image; print('✓ PIL working')"
  ```

---

## ✅ Data Directories

- [ ] Data directory can be created
  ```powershell
  mkdir -Path data/raw -ErrorAction SilentlyContinue
  Test-Path data/raw  # Should return True
  ```

- [ ] Dataset directory can be created
  ```powershell
  mkdir -Path dataset -ErrorAction SilentlyContinue
  Test-Path dataset  # Should return True
  ```

- [ ] Reports directory can be created
  ```powershell
  mkdir -Path reports -ErrorAction SilentlyContinue
  Test-Path reports  # Should return True
  ```

---

## ✅ Source Files

- [ ] `src/collect_dataset.py` exists
  ```powershell
  Test-Path src/collect_dataset.py  # Should return True
  ```

- [ ] `src/split_dataset.py` exists
  ```powershell
  Test-Path src/split_dataset.py  # Should return True
  ```

- [ ] `src/verify_dataset.py` exists
  ```powershell
  Test-Path src/verify_dataset.py  # Should return True
  ```

- [ ] `src/download_data.py` exists (fixed)
  ```powershell
  Test-Path src/download_data.py  # Should return True
  ```

---

## ✅ Test Runs

- [ ] DVC pipeline can be visualized
  ```powershell
  dvc dag
  # Should show 3 stages: collect_images -> split_dataset -> verify_dataset
  ```

- [ ] Configuration can be loaded
  ```powershell
  python -c "
  import yaml
  with open('params.yaml') as f:
      cfg = yaml.safe_load(f)
      print(f'Classes: {len(cfg[\"dataset\"][\"classes\"])}')
  "
  # Should show: Classes: 16
  ```

- [ ] DVC status is clean
  ```powershell
  dvc status
  # Should return nothing or confirm no changes
  ```

---

## ✅ Network Connectivity

- [ ] Can reach DuckDuckGo (test image search)
  ```powershell
  python -c "
  from duckduckgo_search import DDGS
  with DDGS() as ddgs:
      results = list(ddgs.images('test', max_results=1))
      print(f'✓ Connected, found {len(results)} images')
  "
  ```

- [ ] Firewall allows Python to access internet
  - Check Windows Firewall settings
  - Ensure Python is not blocked

---

## ✅ Final Pre-Flight Check

```powershell
# Run this final validation
Write-Host "Running final validation..." -ForegroundColor Cyan

# Check Python
$python = python --version 2>&1
Write-Host "Python: $python" -ForegroundColor Green

# Check DVC
$dvc = dvc version
Write-Host "DVC installed: OK" -ForegroundColor Green

# Check packages
python -c "from duckduckgo_search import DDGS; print('DuckDuckGo: OK')" | Write-Host -ForegroundColor Green
python -c "from PIL import Image; print('Pillow: OK')" | Write-Host -ForegroundColor Green
python -c "import yaml; print('PyYAML: OK')" | Write-Host -ForegroundColor Green

# Check dataset config
python -c "
import yaml
with open('params.yaml') as f:
    cfg = yaml.safe_load(f)
    classes = len(cfg['dataset']['classes'])
    min_img = cfg['dataset']['min_images_per_class']
    print(f'Dataset Config: {classes} politicians, {min_img} min images')
" | Write-Host -ForegroundColor Green

Write-Host ""
Write-Host "✅ All checks passed! Ready to run pipeline." -ForegroundColor Green
Write-Host ""
Write-Host "Next step: dvc repro" -ForegroundColor Yellow
```

---

## 📝 Notes Before Starting

- **Internet required**: Pipeline downloads images from DuckDuckGo. Keep connection stable.

- **Time**: First run takes 15-30 minutes depending on internet speed.

- **Resumable**: If it fails midway, run `dvc repro` again - it resumes from where it left off.

- **Disk space**: Monitor available space. Each politician ~10-20 MB of images.

- **No logs?**: First run downloads images which may appear as "hanging" - it's working. Be patient.

- **Low on images**: If fewer than 80 images collected, run again or adjust `download_images_per_class` in params.yaml

---

## 🚀 Ready to Launch!

Once all checkboxes are marked:

```powershell
dvc repro
```

Monitor output for:
- ✅ "collect_images" stage completion
- ✅ "split_dataset" stage completion  
- ✅ "verify_dataset" stage completion
- ✅ "Dataset verification passed"

Success = Ready for next milestone! 🎉

---

**Date Created**: 2026-05-09  
**Project**: PakPolitician-Classifier (Milestone 1)  
**Classes**: 16 Pakistani Political Figures
