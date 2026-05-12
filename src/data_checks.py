from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]

MIN_PIXELS = 32 * 32


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze():
    report = {
        "per_split_counts": {},
        "corrupt_files": [],
        "small_images": [],
        "duplicates": {},
    }

    md5_map = defaultdict(list)

    for split in SPLITS:
        split_dir = ROOT / split
        counts = {}
        if not split_dir.exists():
            report["per_split_counts"][split] = "missing"
            continue

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            imgs = list(class_dir.glob("**/*.*"))
            counts[class_dir.name] = len(imgs)

            for img in imgs:
                try:
                    with Image.open(img) as im:
                        im.verify()
                    # re-open to get size (verify() can close)
                    with Image.open(img) as im2:
                        w, h = im2.size
                        if w * h < MIN_PIXELS:
                            report["small_images"].append(str(img))
                except Exception as e:
                    report["corrupt_files"].append({"path": str(img), "error": str(e)})
                    continue

                try:
                    md5 = file_md5(img)
                    md5_map[md5].append(str(img))
                except Exception:
                    pass

        report["per_split_counts"][split] = counts

    # duplicates
    for md5, paths in md5_map.items():
        if len(paths) > 1:
            report["duplicates"][md5] = paths

    return report


if __name__ == "__main__":
    import json
    rpt = analyze()
    out = Path("reports")
    out.mkdir(exist_ok=True)
    with open(out / "dataset_check_report.json", "w") as f:
        json.dump(rpt, f, indent=2)
    print("Saved reports/dataset_check_report.json")
