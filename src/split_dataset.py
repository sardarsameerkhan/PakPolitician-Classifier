from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    files.sort()
    return files


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

    summary: dict[str, dict[str, int]] = {}

    for class_name in classes:
        class_raw_dir = raw_root / class_name
        if not class_raw_dir.exists():
            raise FileNotFoundError(f"Class directory missing: {class_raw_dir}")

        images = list_images(class_raw_dir)
        if len(images) < min_images:
            raise RuntimeError(
                f"Class '{class_name}' has {len(images)} images, minimum required is {min_images}."
            )

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
            for img in split_files:
                shutil.copy2(img, target_dir / img.name)

        summary[class_name] = {
            "total": n_total,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

        print(
            f"{class_name}: total={n_total}, train={len(train_files)}, "
            f"val={len(val_files)}, test={len(test_files)}"
        )

    with (report_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Dataset split complete.")


if __name__ == "__main__":
    main()
