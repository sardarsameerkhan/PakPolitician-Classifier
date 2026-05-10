"""
Download REAL politician images from Bing Image Search.

RECOMMENDED METHOD: Automated, respects rate limits, higher quality.

Usage:
    python src/download_bing_images.py
    
OR with custom count:
    python src/download_bing_images.py --count 120
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image
from io import BytesIO
from urllib.parse import urlencode


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def search_bing_images(query: str, max_results: int = 100) -> list[str]:
    """Search for images on Bing."""
    
    urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # Search on Bing
        search_url = f"https://www.bing.com/images/search?{urlencode({'q': query})}"
        response = requests.get(search_url, headers=headers, timeout=10)
        
        # Try to extract image URLs from page
        # Note: This is a simplified approach - for production use bing-image-downloader library
        print(f"    ⚠️  Manual Bing extraction: Use 'pip install bing-image-downloader'")
        
    except Exception as e:
        print(f"    Error searching Bing: {e}")
    
    return urls


def download_image_from_url(url: str, save_path: Path) -> bool:
    """Download and save image from URL."""
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        
        # Try to open and validate image
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Check minimum size
        if img.width < 160 or img.height < 160:
            return False
        
        # Save image
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, format='JPEG', quality=95)
        return True
        
    except Exception as e:
        return False


def install_bing_downloader() -> None:
    """Show instructions to install bing-image-downloader."""
    
    print("\n" + "="*80)
    print("📥 INSTALLING BING IMAGE DOWNLOADER".center(80))
    print("="*80 + "\n")
    
    print("To enable automated image downloading, install:")
    print("\n  pip install bing-image-downloader\n")
    
    print("Then run this script again:")
    print("  python src/download_bing_images.py\n")


def manual_instructions() -> None:
    """Show manual download instructions."""
    
    print("\n" + "="*80)
    print("📥 MANUAL DOWNLOAD INSTRUCTIONS".center(80))
    print("="*80 + "\n")
    
    params = load_params()["dataset"]
    classes = params["classes"]
    
    print("If automated download isn't working, download manually:\n")
    
    for i, politician in enumerate(classes[:3], 1):
        print(f"{i}. {politician.replace('_', ' ').title()}")
        print(f"   - Search: https://www.bing.com/images/search?q={politician.replace('_', '+')}")
        print(f"   - Download 80-100 images")
        print(f"   - Save to: data/raw/{politician}/\n")
    
    print(f"... repeat for remaining {len(classes) - 3} politicians\n")
    
    print("Browser extension option:")
    print("  1. Install: 'Bulk Image Downloader' Chrome extension")
    print("  2. Search on Google Images/Bing for each politician")
    print("  3. Click 'Download all images'")
    print("  4. Save to: data/raw/{politician_name}/\n")


def show_setup() -> None:
    """Show the complete setup process."""
    
    print("\n" + "="*80)
    print("🎬 COMPLETE WORKFLOW".center(80))
    print("="*80 + "\n")
    
    steps = [
        ("Step 1", "Install bing-image-downloader", "pip install bing-image-downloader"),
        ("Step 2", "Download politician images", "python src/download_bing_images.py"),
        ("Step 3", "Split into train/val/test", ".\\venv\\Scripts\\python.exe src/split_dataset.py"),
        ("Step 4", "Verify dataset", ".\\venv\\Scripts\\python.exe src/verify_dataset.py"),
        ("Step 5", "Ready for ML!", "python src/prepare_data.py  # Next step"),
    ]
    
    for title, desc, cmd in steps:
        print(f"{title}: {desc}")
        print(f"   $ {cmd}\n")


def main():
    parser = argparse.ArgumentParser(description='Download politician images from Bing')
    parser.add_argument('--count', type=int, default=100, help='Images per politician')
    parser.add_argument('--no-install', action='store_true', help='Skip install check')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📥 BING IMAGE DOWNLOADER".center(80))
    print("="*80 + "\n")
    
    print("⏳ Checking for bing-image-downloader package...\n")
    
    try:
        # Correct import for bing-image-downloader
        from bing_image_downloader import downloader as bing_downloader
        print("✓ bing-image-downloader is installed\n")
        
        params = load_params()["dataset"]
        classes = params["classes"]
        
        print(f"Starting download for {len(classes)} politicians...")
        print(f"Downloading {args.count} images per politician\n")
        
        raw_root = Path("data") / "raw"
        summary = {}
        
        for idx, politician in enumerate(classes, 1):
            print(f"[{idx:2d}/{len(classes)}] Downloading: {politician.replace('_', ' ').title()}")
            
            output_dir = raw_root / politician
            
            query = politician.replace('_', ' ') + " face portrait"
            
            try:
                # Correct way to call bing_downloader.download()
                bing_downloader.download(
                    query=query,
                    limit=args.count,
                    output_dir=str(output_dir),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=15,
                    verbose=False
                )
                
                # Count downloaded images
                count = len(list(output_dir.glob("*.jpg"))) if output_dir.exists() else 0
                print(f"      ✓ Downloaded: {count} images\n")
                summary[politician] = count
                
            except Exception as e:
                print(f"      ⚠️  Error: {str(e)[:50]}\n")
                summary[politician] = 0
            
            time.sleep(1)  # Rate limiting between politicians
        
        # Show summary
        print("\n" + "="*80)
        print("📊 DOWNLOAD SUMMARY".center(80))
        print("="*80 + "\n")
        
        total = sum(summary.values())
        print(f"Total images downloaded: {total}")
        print(f"\nPer politician:")
        
        for politician, count in summary.items():
            status = "✓" if count >= 80 else "⚠️"
            print(f"  {status} {politician.replace('_', ' ').title():30s} - {count:3d} images")
        
        print("\n" + "="*80)
        print("✅ Download complete!")
        print("\nNext steps:")
        print("  1. python src/split_dataset.py")
        print("  2. python src/verify_dataset.py")
        print("="*80 + "\n")
        
    except ImportError as e:
        print("❌ bing-image-downloader not found\n")
        print("Install it with:")
        print("  pip install bing-image-downloader\n")
        print("Then run this script again:")
        print("  python src/download_bing_images.py\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        print("Make sure:")
        print("  1. bing-image-downloader is installed: pip install bing-image-downloader")
        print("  2. Internet connection is working")
        print("  3. Try again: python src/download_bing_images.py\n")


if __name__ == "__main__":
    main()
