from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    files.sort(key=lambda path: path.relative_to(folder).as_posix().lower())
    return files


def save_numbered_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=95)


def main() -> None:
    dataset_cfg = load_params()["dataset"]

    classes: list[str] = dataset_cfg["classes"]
    min_images = int(dataset_cfg["min_images_per_class"])
    random_state = int(dataset_cfg["random_state"])

    split_cfg = dataset_cfg["split"]
    train_ratio = float(split_cfg["train"])
    val_ratio = float(split_cfg["val"])
    test_ratio = float(split_cfg["test"])

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    raw_root = Path("data") / "raw"
    out_root = Path("dataset")
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split_dir = out_root / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    random.seed(random_state)

    summary: dict[str, dict[str, Any]] = {}
    skipped_classes: list[str] = []

    for class_name in classes:
        class_raw_dir = raw_root / class_name
        if not class_raw_dir.exists():
            print(f"WARNING: Class directory missing, skipping: {class_raw_dir}")
            skipped_classes.append(class_name)
            continue

        images = list_images(class_raw_dir)
        if len(images) < min_images:
            print(
                f"WARNING: Class '{class_name}' has {len(images)} images, minimum required is {min_images}. Skipping."
            )
            skipped_classes.append(class_name)
            summary[class_name] = {
                "total": len(images),
                "train": 0,
                "val": 0,
                "test": 0,
                "skipped": True,
            }
            continue

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train : n_train + n_val]
        test_files = images[n_train + n_val :]

        split_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, split_files in split_map.items():
            target_dir = out_root / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, img in enumerate(split_files, 1):
                target_name = f"{class_name}_{index:03d}.jpg"
                save_numbered_image(img, target_dir / target_name)

        summary[class_name] = {
            "total": n_total,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "skipped": False,
        }

        print(
            f"{class_name}: total={n_total}, train={len(train_files)}, "
            f"val={len(val_files)}, test={len(test_files)}"
        )

    with (report_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if skipped_classes:
        print(f"Skipped classes: {', '.join(skipped_classes)}")

    print("Dataset split complete.")


if __name__ == "__main__":
    main()
