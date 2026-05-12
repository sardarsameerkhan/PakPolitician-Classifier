from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.relative_to(folder).as_posix().lower())
    return files


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_class(
    class_name: str,
    class_src: Path,
    class_out: Path,
    min_width: int,
    min_height: int,
) -> dict[str, Any]:
    class_out.mkdir(parents=True, exist_ok=True)

    images = list_images(class_src)
    seen_hashes: set[str] = set()

    stats: dict[str, Any] = {
        "input": len(images),
        "saved": 0,
        "corrupt": 0,
        "too_small": 0,
        "duplicates": 0,
        "errors": [],
    }

    out_index = 1

    for image_path in images:
        try:
            img_hash = file_md5(image_path)
            if img_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(img_hash)

            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                width, height = img.size
                if width < min_width or height < min_height:
                    stats["too_small"] += 1
                    continue

                output_name = f"{class_name}_{out_index:04d}.jpg"
                output_path = class_out / output_name
                img.save(output_path, format="JPEG", quality=95)
                out_index += 1
                stats["saved"] += 1
        except Exception as exc:
            stats["corrupt"] += 1
            stats["errors"].append({"path": str(image_path), "error": str(exc)})

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw politician images")
    parser.add_argument("--source-root", default="data/raw", help="Source dataset root")
    parser.add_argument("--output-root", default="data/processed", help="Output dataset root")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output root before preprocessing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_params()["dataset"]

    classes: list[str] = cfg["classes"]
    min_images = int(cfg["min_images_per_class"])
    min_width = int(cfg["min_width"])
    min_height = int(cfg["min_height"])

    src_root = Path(args.source_root)
    out_root = Path(args.output_root)
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    if args.clean_output and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "min_width": min_width,
        "min_height": min_height,
        "min_images_per_class": min_images,
        "classes": {},
        "skipped_classes": [],
    }

    for class_name in classes:
        class_src = src_root / class_name
        class_out = out_root / class_name

        if not class_src.exists():
            print(f"WARNING: Missing class folder: {class_src}")
            summary["skipped_classes"].append(class_name)
            summary["classes"][class_name] = {
                "input": 0,
                "saved": 0,
                "corrupt": 0,
                "too_small": 0,
                "duplicates": 0,
                "meets_minimum": False,
                "missing_source": True,
            }
            continue

        stats = preprocess_class(class_name, class_src, class_out, min_width, min_height)
        stats["meets_minimum"] = stats["saved"] >= min_images
        summary["classes"][class_name] = stats

        if not stats["meets_minimum"]:
            summary["skipped_classes"].append(class_name)

        print(
            f"{class_name}: input={stats['input']}, saved={stats['saved']}, "
            f"small={stats['too_small']}, dup={stats['duplicates']}, corrupt={stats['corrupt']}"
        )

    with (report_root / "preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Preprocessing complete")
    print("[OK] Processed images written to", out_root)
    print("[OK] Summary report written to reports/preprocess_summary.json")


if __name__ == "__main__":
    main()
