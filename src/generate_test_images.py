"""
Generate sample images for testing the dataset pipeline.
This creates synthetic images locally to validate the pipeline works.
Later, replace with real images from actual sources.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_sample_face(politician_name: str, image_num: int) -> Image.Image:
    """Create a synthetic face-like image for testing."""
    # Create face-like pattern
    width, height = 200, 250
    
    # Use politician name for deterministic colors
    random.seed(hash(politician_name + str(image_num)) % 2**32)
    
    # Background color variation
    bg_r = random.randint(200, 256)
    bg_g = random.randint(200, 256)
    bg_b = random.randint(200, 256)
    bg_color = (bg_r, bg_g, bg_b)
    
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw simple face shape (oval)
    face_r = random.randint(150, 200)
    face_g = random.randint(150, 200)
    face_b = random.randint(150, 200)
    face_color = (face_r, face_g, face_b)
    
    draw.ellipse([30, 20, 170, 220], fill=face_color, outline=(100, 100, 100), width=2)
    
    # Draw eyes
    eye_color = (0, 0, 0)
    draw.ellipse([60, 80, 80, 100], fill=eye_color)
    draw.ellipse([120, 80, 140, 100], fill=eye_color)
    
    # Draw nose
    draw.polygon([(100, 100), (95, 130), (105, 130)], fill=eye_color)
    
    # Draw mouth
    draw.arc([70, 140, 130, 190], 0, 180, fill=eye_color, width=2)
    
    # Add some random dots for texture
    for _ in range(30):
        x = random.randint(0, width)
        y = random.randint(0, height)
        dot_r = random.randint(0, 256)
        dot_g = random.randint(0, 256)
        dot_b = random.randint(0, 256)
        color = (dot_r, dot_g, dot_b)
        draw.point((x, y), fill=color)
    
    return img


def main() -> None:
    params = load_params()["dataset"]
    
    classes: list[str] = params["classes"]
    min_images = int(params["min_images_per_class"])
    
    raw_root = Path("data") / "raw"
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating sample images for testing...")
    print(f"   Classes: {len(classes)} politicians")
    print(f"   Images per class: {min_images}")
    
    summary: dict[str, dict[str, int]] = {}
    
    for class_idx, class_name in enumerate(classes):
        class_dir = raw_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        existing = list(class_dir.glob("*.jpg"))
        start_num = len(existing)
        
        for i in range(start_num, min_images):
            img = create_sample_face(class_name, i)
            img_path = class_dir / f"sample_{i:03d}.jpg"
            img.save(img_path, format="JPEG", quality=95)
        
        total = len(list(class_dir.glob("*.jpg")))
        summary[class_name] = {
            "generated": min_images,
            "total": total,
        }
        
        pct = int((class_idx + 1) / len(classes) * 100)
        print(f"   [{pct:3d}%] ✓ {class_name.replace('_', ' ').title():30s} - {total} images")
    
    # Save summary
    with (report_root / "download_summary.json").open("w") as f:
        json.dump({
            "note": "Generated test images - replace with real data",
            "classes": summary,
        }, f, indent=2)
    
    print(f"\n✅ Generated {len(classes)} × {min_images} = {len(classes) * min_images} test images")
    print(f"   Location: data/raw/")
    print(f"\n⚠️  NOTE: These are SYNTHETIC test images.")
    print(f"   For production, replace with real politician face images from:")
    print(f"   - Bing Image Search")
    print(f"   - Google Images")
    print(f"   - Manually sourced images")
    print(f"\n   Pipeline validation: Ready ✓")


if __name__ == "__main__":
    main()
