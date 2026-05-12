from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path("dataset")
BACKUP_ROOT = Path("dataset_duplicates_backup") / "ambiguous"
REPORT_PATH = Path("reports") / "ambiguous_duplicates_removed.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images() -> list[Path]:
    paths: list[Path] = []
    for split in ("train", "val", "test"):
        split_dir = ROOT / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                paths.append(path)
    return paths


def main() -> None:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for img_path in iter_images():
        try:
            with Image.open(img_path) as img:
                img.verify()
            md5 = file_md5(img_path)
            by_hash[md5].append(img_path)
        except Exception as exc:
            print(f"Skipping unreadable image: {img_path} ({exc})")

    removed: list[dict[str, str]] = []

    for md5, paths in by_hash.items():
        classes = {p.parent.name for p in paths}
        if len(classes) <= 1:
            continue

        # Exact same pixels appear under multiple labels, so remove all copies.
        for path in paths:
            rel = path.relative_to(ROOT)
            dest = BACKUP_ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                shutil.move(str(path), str(dest))
                removed.append(
                    {
                        "md5": md5,
                        "path": str(path),
                        "backup": str(dest),
                        "class_name": path.parent.name,
                    }
                )

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(removed, f, indent=2)

    print(f"Removed {len(removed)} ambiguous duplicates")
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
