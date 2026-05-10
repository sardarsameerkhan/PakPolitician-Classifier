from __future__ import annotations

import hashlib
import json
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image, UnidentifiedImageError

try:
    from ddgs import DDGS
    USING_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        USING_DDGS = False
    except ImportError:
        print("ERROR: Neither 'ddgs' nor 'duckduckgo_search' found. Run: pip install ddgs")
        raise


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_save_image(content: bytes, output_path: Path, min_width: int, min_height: int) -> bool:
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return False

    if img.width < min_width or img.height < min_height:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="JPEG", quality=95)
    return True


def image_urls(query: str, max_results: int) -> list[str]:
    urls: list[str] = []
    retries = 5
    for attempt in range(retries):
        try:
            ddgs = DDGS()
            results = list(ddgs.images(keywords=query, max_results=max_results))
            for item in results:
                url = item.get("image")
                if isinstance(url, str) and url.startswith("http"):
                    urls.append(url)
            break
        except Exception as e:
            wait_time = 15 + (attempt * 10)
            print(f"    Attempt {attempt + 1}/{retries} failed ({type(e).__name__}). Waiting {wait_time}s...")
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                print(f"    ⚠️  Could not fetch URLs after {retries} attempts")
    return urls


def main() -> None:
    params = load_params()["dataset"]

    classes: list[str] = params["classes"]
    class_queries: dict[str, str] = params["class_queries"]
    min_images = int(params["min_images_per_class"])
    target_download = int(params["download_images_per_class"])
    min_width = int(params["min_width"])
    min_height = int(params["min_height"])

    random.seed(int(params["random_state"]))

    raw_root = Path("data") / "raw"
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
    )

    summary: dict[str, dict[str, int]] = {}

    for idx, class_name in enumerate(classes):
        if idx > 0:
            print(f"\n⏳ Waiting 30 seconds before next politician (rate limit)...")
            time.sleep(30)
            
        print(f"\n🔍 Collecting images for: {class_name.replace('_', ' ').title()}")
        class_dir = raw_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        existing = list(class_dir.glob("*.jpg"))
        saved_count = len(existing)
        seen_hashes = set()

        for path in existing:
            seen_hashes.add(path.stem)

        if saved_count > 0:
            print(f"   Found {saved_count} existing images, continuing...")

        query = class_queries.get(class_name, class_name.replace("_", " "))
        print(f"   Query: {query}")
        urls = image_urls(query, max_results=max(target_download * 3, 300))
        print(f"   Found {len(urls)} URLs to check")
        
        if not urls:
            print(f"   ⚠️  No URLs found - likely rate limited")
            summary[class_name] = {
                "required_minimum": min_images,
                "target_download": target_download,
                "downloaded_this_run": 0,
                "total_available": saved_count,
                "skipped": 0,
            }
            continue
            
        random.shuffle(urls)

        downloaded = 0
        skipped = 0

        for url_idx, url in enumerate(urls):
            if saved_count >= target_download:
                break

            try:
                response = session.get(url, timeout=12)
                response.raise_for_status()
                content = response.content
            except requests.RequestException as e:
                skipped += 1
                continue

            digest = hashlib.sha1(content).hexdigest()
            if digest in seen_hashes:
                skipped += 1
                continue

            out_path = class_dir / f"{digest}.jpg"
            ok = safe_save_image(content, out_path, min_width=min_width, min_height=min_height)
            if not ok:
                skipped += 1
                continue

            seen_hashes.add(digest)
            saved_count += 1
            downloaded += 1

            # More respectful delays
            if downloaded % 5 == 0:
                print(f"   Downloaded: {downloaded}/{target_download}", end="\r")
            time.sleep(0.5)

        print(f"   ✓ {class_name}: total={saved_count}, new={downloaded}, skipped={skipped}      ")

        summary[class_name] = {
            "required_minimum": min_images,
            "target_download": target_download,
            "downloaded_this_run": downloaded,
            "total_available": saved_count,
            "skipped": skipped,
        }

    with (report_root / "download_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    failed = [name for name, info in summary.items() if info["total_available"] < min_images]
    if failed:
        names = ", ".join(failed)
        print(f"\n⚠️  WARNING: Minimum image requirement not met for: {names}")
        print(f"     This may be due to rate limiting. Try running again later.")
        print(f"     Failed classes: {names}")
        # Don't fail hard - allow partial dataset for now
        # raise RuntimeError(f"Minimum image requirement not met for: {names}")

    print("\n✅ Image collection completed.")
    print(f"\nSummary saved to: reports/download_summary.json")


if __name__ == "__main__":
    main()
