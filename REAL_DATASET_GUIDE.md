# 🖼️ Real Dataset Collection - Step-by-Step Guide

## Your Best Options

### **Option 1: AUTOMATIC - Bing Image Downloader (Recommended)**

**Fastest & Easiest way to get real images!**

```powershell
# Step 1: Install the downloader
pip install bing-image-downloader

# Step 2: Run this command to download all images automatically
python src/download_bing_images.py

# Step 3: Split and verify
python src/split_dataset.py
python src/verify_dataset.py
```

**⏱️ Time: ~30-45 minutes for all 16 politicians**

---

### **Option 2: MANUAL - Google Images or Bing**

**If automated isn't working**

1. **Open browser** → Go to: https://www.bing.com/images
   
2. **For each politician**, search:
   - "Imran Khan face"
   - "Nawaz Sharif portrait"
   - etc.
   
3. **Download images**:
   - Select 80-100 images
   - Save to: `data/raw/{politician_name}/`
   
4. **Repeat for all 16 politicians**

5. **Then run**:
   ```powershell
   python src/split_dataset.py
   python src/verify_dataset.py
   ```

**⏱️ Time: ~2-3 hours manually downloading**

---

### **Option 3: Official Government Sources**

**High quality, authorized images**

Visit and download from:
- Pakistan's National Assembly website
- PM Office official website
- Provincial government websites
- Official news agency (APP, PPI)

Place images in: `data/raw/{politician_name}/`

---

## Directory Structure Required

```
data/raw/
├── imran_khan/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ... (80+ images per politician)
├── nawaz_sharif/
│   └── ... (80+ images)
├── shehbaz_sharif/
│   └── ... (80+ images)
... repeat for all 16 ...
└── ahmed_sharif_chaudhry/
    └── ... (80+ images)
```

**⚠️ Important**: Folder names MUST MATCH exactly - use underscores between words!

---

## After Downloading - Processing Steps

Once you have images in `data/raw/`:

```powershell
# Step 1: Split images into train/val/test
python src/split_dataset.py

# Step 2: Verify everything is correct
python src/verify_dataset.py

# Step 3: Check the report
Get-Content reports/dataset_report.json | ConvertFrom-Json

# Expected output: "valid": true
```

---

## ⚠️ Image Quality Tips

### GOOD images ✓
- Clear face visible
- Professional quality
- Front or 3/4 angle
- Minimum 160×160 pixels
- One face per image

### BAD images ✗
- Group photos (multiple people)
- Face obscured/covered
- Too small
- Blurry or low quality
- Not actually the politician

---

## 🚀 Recommended Workflow

### FASTEST (1 hour total):
```powershell
# 1. Install and download automatically
pip install bing-image-downloader
python src/download_bing_images.py

# 2. Process automatically
python src/split_dataset.py
python src/verify_dataset.py

# 3. Ready for ML!
✅ Dataset complete
```

### MANUAL (2-3 hours):
```powershell
# 1. Download manually from Bing/Google for each politician
# 2. Place 80+ images per politician in data/raw/{politician_name}/
# 3. Process
python src/split_dataset.py
python src/verify_dataset.py

# 4. Ready for ML!
✅ Dataset complete
```

---

## 📊 Dataset Statistics After Processing

Once you run the split & verify scripts:

- ✅ Training: 960 images (75%)
- ✅ Validation: 192 images (15%)
- ✅ Testing: 128 images (10%)
- ✅ All 16 politicians covered
- ✅ Clean, organized structure

---

## Commands Quick Reference

```powershell
# Install Bing downloader
pip install bing-image-downloader

# Auto-download images
python src/download_bing_images.py

# Manual: Count what you have
Get-ChildItem data/raw -Recurse -File | Measure-Object

# Split images
python src/split_dataset.py

# Verify
python src/verify_dataset.py

# View report
Get-Content reports/dataset_report.json | ConvertFrom-Json

# Check specific politician
Get-ChildItem dataset/train/imran_khan | Measure-Object
```

---

## Next Steps After Real Dataset is Ready

Once `dataset_report.json` shows `"valid": true`:

1. **Feature Extraction** - Convert images to embeddings
2. **Model Training** - Train classifier
3. **Evaluation** - Test performance

---

## 💡 Pro Tips

✓ Start with **Option 1** (automatic) - if it works, saves hours!  
✓ Use **high-quality** images from news/official sources  
✓ **Variety matters** - different angles, lighting, expressions help  
✓ **One face per image** - crops if needed  
✓ **Respect copyrights** - use public domain or properly licensed images  
✓ **Minimum 80 images** per politician, **target 100-120** for better results  

---

**Status**: Your folders are ready to receive real images!  
**Next**: Choose your download method above and start collecting! 🖼️
