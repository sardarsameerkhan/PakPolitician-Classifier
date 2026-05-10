from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    cfg = load_params()["dataset"]
    classes: list[str] = cfg["classes"]
    min_images = int(cfg["min_images_per_class"])

    root = Path("dataset")
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "minimum_images_per_class": min_images,
        "classes": {},
        "valid": True,
    }

    errors: list[str] = []

    for class_name in classes:
        train_count = count_images(root / "train" / class_name)
        val_count = count_images(root / "val" / class_name)
        test_count = count_images(root / "test" / class_name)
        total = train_count + val_count + test_count

        class_ok = total >= min_images and train_count > 0 and val_count > 0 and test_count > 0
        if not class_ok:
            errors.append(
                f"{class_name}: total={total}, train={train_count}, val={val_count}, test={test_count}"
            )

        report["classes"][class_name] = {
            "total": total,
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "meets_minimum": total >= min_images,
            "has_all_splits": train_count > 0 and val_count > 0 and test_count > 0,
        }

    if errors:
        report["valid"] = False
        report["errors"] = errors

    with (report_root / "dataset_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not report["valid"]:
        joined = "\n".join(errors)
        print(f"Dataset verification found incomplete classes:\n{joined}")
    else:
        print("Dataset verification passed.")


if __name__ == "__main__":
    main()
