"""
Download REAL politician images for the dataset.

This script provides options to collect real images from legitimate sources.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def show_options() -> None:
    """Show options for getting real images."""
    
    print("\n" + "="*80)
    print("🖼️  REAL DATASET OPTIONS - Choose Your Source".center(80))
    print("="*80 + "\n")
    
    options = {
        "1": {
            "name": "✅ Manually Download Images",
            "description": "Download from websites and place in folders",
            "sources": [
                "Wikipedia politician pages",
                "Official government websites",
                "News agency websites (Dawn, Express, etc.)",
                "News24/Geo/ARY news archives"
            ],
            "steps": [
                "Find politician images online",
                "Save images in: data/raw/{politician_name}/",
                "Run: python src/split_dataset.py",
                "Run: python src/verify_dataset.py"
            ]
        },
        "2": {
            "name": "🔗 Bing Image Search (Recommended)",
            "description": "Automated download from Bing",
            "sources": ["Bing Image Search API"],
            "steps": [
                "pip install bing-image-downloader",
                "Use: bing_image_downloader.py script (coming)",
                "Downloads ~100 images per politician",
                "Automatically saves to correct folders"
            ]
        },
        "3": {
            "name": "📸 Google Images (Manual)",
            "description": "Download from Google Images manually",
            "sources": ["Google Images"],
            "steps": [
                "Search: '[Politician Name] face'",
                "Bulk download using browser extension",
                "Save to: data/raw/{politician_name}/",
                "Run split and verify scripts"
            ]
        },
        "4": {
            "name": "🏛️  Official Sources",
            "description": "Download from official government/news sources",
            "sources": [
                "Pakistan's Parliament website",
                "PM Office official photos",
                "Provincial government websites",
                "Official news agency photos (APP, PPI)"
            ],
            "steps": [
                "Visit official government websites",
                "Download high-quality politician portraits",
                "Save to: data/raw/{politician_name}/",
                "Run split and verify scripts"
            ]
        },
        "5": {
            "name": "🎬 Video Frame Extraction",
            "description": "Extract frames from news videos/talks",
            "sources": ["YouTube news clips", "TV interviews"],
            "steps": [
                "Record/download political news/talks",
                "Extract clean frames with politician faces",
                "Save frames as JPEG images",
                "Place in: data/raw/{politician_name}/"
            ]
        }
    }
    
    for num, info in options.items():
        print(f"\n{info['name']}")
        print(f"{'─' * 75}")
        print(f"Description: {info['description']}")
        print(f"\n📍 Sources:")
        for source in info['sources']:
            print(f"   • {source}")
        print(f"\n📝 Steps:")
        for i, step in enumerate(info['steps'], 1):
            print(f"   {i}. {step}")
    
    print("\n" + "="*80)


def show_directory_structure() -> None:
    """Show where to place images."""
    
    params = load_params()["dataset"]
    classes = params["classes"]
    
    print("\n" + "="*80)
    print("📁 WHERE TO PLACE YOUR IMAGES".center(80))
    print("="*80 + "\n")
    
    print("Create folders and download images here:")
    print(f"\ndata/raw/")
    
    for politician in classes[:3]:  # Show first 3 as examples
        print(f"├── {politician}/")
        print(f"│   ├── image_1.jpg")
        print(f"│   ├── image_2.jpg")
        print(f"│   └── ... (80+ images needed)")
    
    print(f"├── ... ({len(classes) - 3} more politicians)")
    print(f"└── ahmed_sharif_chaudhry/")
    print(f"    ├── image_1.jpg")
    print(f"    └── ... (80+ images needed)")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"   • Create EXACTLY these folder names (must match params.yaml)")
    print(f"   • Each politician needs minimum 80 images")
    print(f"   • Images should be clear face portraits (JPG/PNG)")
    print(f"   • Minimum size: 160×160 pixels")
    print(f"   • One face per image (for better results)")


def show_next_steps() -> None:
    """Show what to do after downloading images."""
    
    print("\n" + "="*80)
    print("🚀 WHAT TO DO AFTER DOWNLOADING IMAGES".center(80))
    print("="*80 + "\n")
    
    print("After you place real images in data/raw/:\n")
    
    steps = [
        ("Step 1", "Verify folders are created", "ls data/raw/"),
        ("Step 2", "Count images per politician", "Get-ChildItem data/raw -Recurse | Measure-Object"),
        ("Step 3", "Split into train/val/test", ".\\venv\\Scripts\\python.exe src/split_dataset.py"),
        ("Step 4", "Verify dataset", ".\\venv\\Scripts\\python.exe src/verify_dataset.py"),
        ("Step 5", "Check report", "Get-Content reports/dataset_report.json | ConvertFrom-Json"),
    ]
    
    for title, desc, cmd in steps:
        print(f"{title}: {desc}")
        print(f"   Command: {cmd}\n")
    
    print("✅ Once verified, your real dataset is ready for:")
    print("   • Feature extraction")
    print("   • Model training")
    print("   • Evaluation")


def check_existing_images() -> dict[str, int]:
    """Check how many real images exist currently."""
    
    params = load_params()["dataset"]
    classes = params["classes"]
    raw_root = Path("data") / "raw"
    
    counts = {}
    for politician in classes:
        politician_dir = raw_root / politician
        if politician_dir.exists():
            count = len(list(politician_dir.glob("*.jpg"))) + len(list(politician_dir.glob("*.png")))
            counts[politician] = count
        else:
            counts[politician] = 0
    
    return counts


def show_current_status() -> None:
    """Show current dataset status."""
    
    print("\n" + "="*80)
    print("📊 CURRENT DATASET STATUS".center(80))
    print("="*80 + "\n")
    
    counts = check_existing_images()
    total = sum(counts.values())
    
    print(f"Total images collected: {total}\n")
    
    params = load_params()["dataset"]
    min_images = params["min_images_per_class"]
    
    ready = [p for p, c in counts.items() if c >= min_images]
    needs = [p for p, c in counts.items() if c < min_images]
    
    if ready:
        print(f"✓ Ready ({len(ready)}):")
        for politician in ready[:5]:
            print(f"  • {politician}: {counts[politician]} images")
        if len(ready) > 5:
            print(f"  ... and {len(ready) - 5} more")
    
    if needs:
        print(f"\n⚠️  Need more images ({len(needs)}):")
        for politician in needs[:5]:
            needed = min_images - counts[politician]
            print(f"  • {politician}: {counts[politician]}/{min_images} (need {needed} more)")
        if len(needs) > 5:
            print(f"  ... and {len(needs) - 5} more")


def main() -> None:
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + "🖼️  REAL DATASET COLLECTION GUIDE".center(78) + "║")
    print("║" + "Pakistani Politician Face Classification".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    show_current_status()
    show_options()
    show_directory_structure()
    show_next_steps()
    
    print("\n" + "="*80)
    print("💡 TIPS FOR BEST RESULTS".center(80))
    print("="*80 + "\n")
    
    tips = [
        "Use HIGH QUALITY images (clear faces, professional photos)",
        "Prefer frontal or 3/4 angle face shots",
        "Avoid group photos - one face per image",
        "Remove duplicates manually to save storage",
        "Respect copyright - give attribution where needed",
        "Use images from public/news sources",
        "Ensure faces are clearly visible (not covered)",
        "Various poses/angles help model generalization"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i:2d}. {tip}")
    
    print("\n" + "="*80)
    print("📞 QUICK COMMANDS".center(80))
    print("="*80 + "\n")
    
    commands = {
        "Create folder structure": "mkdir data/raw; mkdir data/raw/{imran_khan,nawaz_sharif,...}",
        "Count images per politician": "Get-ChildItem data/raw | ForEach-Object { Write-Host $_.Name; (Get-ChildItem $_.FullName).Count }",
        "Split your images": ".\\venv\\Scripts\\python.exe src/split_dataset.py",
        "Verify dataset": ".\\venv\\Scripts\\python.exe src/verify_dataset.py",
        "Check report": "Get-Content reports/dataset_report.json | ConvertFrom-Json"
    }
    
    for desc, cmd in commands.items():
        print(f"{desc}:")
        print(f"  → {cmd}\n")
    
    print("="*80)
    print("✅ Ready to download? Start with Option 1 or Option 2!".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
