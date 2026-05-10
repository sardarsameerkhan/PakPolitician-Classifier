#!/usr/bin/env python3
"""
Download REAL politician face images using multiple sources
with improved search queries specifically targeting face portraits
"""

import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def download_with_retry(url, timeout=10, max_retries=3):
    """Download image with retry logic"""
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                return None
                
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    return None

def validate_image(image_data, min_size=160):
    """Validate that image is actually a face-like image"""
    try:
        img = Image.open(BytesIO(image_data)).convert('RGB')

        # Check dimensions
        width, height = img.size
        if width < min_size or height < min_size:
            return False

        # Convert to numpy array for face detection
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(int(min_size/4), int(min_size/4)))
        return len(faces) > 0
    except Exception:
        return False

def save_image(image_data, output_path):
    """Save and convert image to JPEG"""
    try:
        img = Image.open(BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception:
        return False

def download_politician_images(politician_name, output_dir, count=100, 
                              search_queries=None):
    """Download images for a politician with improved queries"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default queries if not provided
    if not search_queries:
        search_queries = [f"{politician_name} face portrait"]
    
    print(f"\n📥 {politician_name.title().replace('_', ' ')}")
    print(f"   Output: {output_dir}")
    
    saved_count = 0
    
    # Use Bing downloader for real images, save to temp then validate faces
    import shutil
    try:
        from bing_image_downloader import downloader

        temp_root = Path("temp_bing") / politician_name

        for query in search_queries:
            if saved_count >= count:
                break

            try:
                print(f"   Searching: {query}...")

                # prepare temp folder
                if temp_root.exists():
                    shutil.rmtree(temp_root)
                temp_root.mkdir(parents=True, exist_ok=True)

                # Download into temp_root; downloader creates a subfolder named after the query
                downloader.download(
                    query=query,
                    limit=min(200, count * 2),
                    output_dir=str(temp_root),
                    adult_filter_off=True,
                    force_replace=True,
                    timeout=15,
                    verbose=False
                )

                # find the subfolder created (usually same as query text)
                subfolders = [p for p in temp_root.iterdir() if p.is_dir()]
                if not subfolders:
                    print("   ⚠️  No images downloaded by downloader for this query")
                    continue
                query_dir = subfolders[0]

                # Validate and move images
                for img_file in query_dir.glob('*'):
                    if saved_count >= count:
                        break
                    try:
                        data = img_file.read_bytes()
                        if not validate_image(data):
                            continue
                        # save to final output_dir with unique name
                        fname = f"{politician_name}_{saved_count+1:04d}.jpg"
                        out_path = output_dir / fname
                        save_image(data, str(out_path))
                        saved_count += 1
                    except Exception:
                        continue

                print(f"   ✓ {saved_count} images saved so far")

            except Exception as e:
                print(f"   ⚠️  Query failed: {str(e)[:120]}")
                continue

            # cleanup and rate limit
            if temp_root.exists():
                shutil.rmtree(temp_root)
            time.sleep(2)

    except ImportError:
        print("   ⚠️  bing-image-downloader not available")
        return 0
    
    return saved_count

def main():
    parser = argparse.ArgumentParser(description='Download real politician images')
    parser.add_argument('--count', type=int, default=100,
                       help='Images per politician (default: 100)')
    parser.add_argument('--params', type=str, default='params.yaml',
                       help='Path to params.yaml')
    args = parser.parse_args()
    
    # Load parameters
    import yaml
    try:
        with open(args.params, 'r') as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading params.yaml: {e}")
        return 1
    
    # params.yaml uses dataset.classes and dataset.class_queries
    dataset_cfg = params.get('dataset', {})
    politicians = dataset_cfg.get('classes', [])
    class_queries = dataset_cfg.get('class_queries', {})
    if not politicians:
        print("No politicians found in params.yaml under dataset.classes")
        return 1
    
    print("\n" + "="*80)
    print("         🇵🇰 REAL POLITICIAN IMAGE DOWNLOADER 🇵🇰")
    print("="*80)
    print(f"Downloading {args.count} images per politician ({len(politicians)} total)\n")
    
    data_raw_dir = Path("data/raw")
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    total_downloaded = 0
    
    for idx, politician in enumerate(politicians, 1):
        print(f"\n[{idx}/{len(politicians)}] Downloading: {politician.replace('_', ' ').title()}")
        
        output_dir = data_raw_dir / politician
        
        # Enhanced search queries for better results; prefer custom query if available
        custom = class_queries.get(politician)
        if custom:
            search_queries = [custom, f"{politician.replace('_', ' ')} face", f"{politician.replace('_', ' ')} portrait"]
        else:
            search_queries = [
                f"{politician.replace('_', ' ')} politician",
                f"{politician.replace('_', ' ')} face",
                f"{politician.replace('_', ' ')} photo",
                f"{politician.replace('_', ' ')} portrait",
            ]
        
        count = download_politician_images(
            politician,
            output_dir,
            count=args.count,
            search_queries=search_queries
        )
        
        total_downloaded += count
        
        # Rate limiting between politicians
        if idx < len(politicians):
            print(f"   ⏳ Rate limiting (2 seconds)...")
            time.sleep(2)
    
    print("\n" + "="*80)
    print(f"✅ DOWNLOAD COMPLETE")
    print(f"   Total images downloaded: {total_downloaded}")
    print(f"   Target: {len(politicians) * args.count}")
    print("="*80)
    
    # Check what we got
    print("\n📊 IMAGES PER POLITICIAN:")
    for politician in politicians:
        output_dir = data_raw_dir / politician
        if output_dir.exists():
            count = len(list(output_dir.glob("*.jpg"))) + len(list(output_dir.glob("*.png")))
            status = "✓" if count >= 80 else "⚠️"
            print(f"   {status} {politician:30} : {count:3d} images")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
