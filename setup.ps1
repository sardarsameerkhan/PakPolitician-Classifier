# Quick Start Script for PakPolitician-Classifier
# Run this from PowerShell in the project root directory

Write-Host "========================================" -ForegroundColor Green
Write-Host "PakPolitician-Classifier - Quick Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check Python installation
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Cyan
$python_check = python --version 2>&1
if ($python_check -like "Python*") {
    Write-Host "✓ Python found: $python_check" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check/Create virtual environment
Write-Host ""
Write-Host "[2/4] Setting up virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Install requirements
Write-Host ""
Write-Host "[3/4] Installing dependencies..." -ForegroundColor Cyan
pip install -q --upgrade pip
pip install -q -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Initialize DVC
Write-Host ""
Write-Host "[4/4] Initializing DVC..." -ForegroundColor Cyan
if (-not (Test-Path ".\.dvc")) {
    dvc init -q
    Write-Host "✓ DVC initialized" -ForegroundColor Green
} else {
    Write-Host "✓ DVC already initialized" -ForegroundColor Green
}

# Setup complete
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Dataset Configuration:" -ForegroundColor Yellow
Write-Host "  • 16 Pakistani political figures" -ForegroundColor White
Write-Host "  • 80-140 images per politician" -ForegroundColor White
Write-Host "  • Train/Val/Test split: 75/15/10" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Start Data Collection:" -ForegroundColor White
Write-Host "   dvc repro" -ForegroundColor Cyan
Write-Host ""
Write-Host "   OR run individual steps:" -ForegroundColor White
Write-Host "   python src/collect_dataset.py" -ForegroundColor Cyan
Write-Host "   python src/split_dataset.py" -ForegroundColor Cyan
Write-Host "   python src/verify_dataset.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Check Results:" -ForegroundColor White
Write-Host "   Get-Content reports/dataset_report.json | ConvertFrom-Json" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Read Setup Guide:" -ForegroundColor White
Write-Host "   SETUP_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "Expected Execution Time: 15-30 minutes (depending on internet)" -ForegroundColor Yellow
Write-Host ""
