from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_params() -> dict[str, Any]:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.relative_to(folder).as_posix().lower())
    return files


def build_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
    return detector


def detect_largest_face(detector: cv2.CascadeClassifier, image_bgr) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def expand_box(x: int, y: int, w: int, h: int, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(width, x + w + mx)
    y2 = min(height, y + h + my)
    return x1, y1, x2, y2


def save_jpeg(img_bgr, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract/crop faces from class image folders")
    parser.add_argument("--source-root", default="data/processed", help="Input root containing class folders")
    parser.add_argument("--output-root", default="data/faces", help="Output root for face images")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional class names to process (default: all classes from params.yaml)",
    )
    parser.add_argument("--margin", type=float, default=0.35, help="Relative margin around detected face")
    parser.add_argument(
        "--drop-no-face",
        action="store_true",
        help="Drop images where no face is detected (default keeps original image)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output root before processing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_params()["dataset"]
    cfg_classes: list[str] = cfg["classes"]
    classes: list[str] = args.classes if args.classes else cfg_classes

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        fallback = Path("data") / "raw"
        if fallback.exists():
            source_root = fallback
        else:
            raise FileNotFoundError(f"Source root not found: {source_root}")

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    detector = build_detector()

    summary: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "drop_no_face": bool(args.drop_no_face),
        "margin": float(args.margin),
        "classes": {},
    }

    for class_name in classes:
        class_src = source_root / class_name
        class_out = output_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        stats = {
            "input": 0,
            "saved": 0,
            "face_detected": 0,
            "no_face": 0,
            "dropped_no_face": 0,
            "errors": 0,
        }

        if not class_src.exists():
            summary["classes"][class_name] = stats
            continue

        images = list_images(class_src)
        stats["input"] = len(images)

        out_index = 1
        for image_path in images:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    stats["errors"] += 1
                    continue

                h, w = image.shape[:2]
                bbox = detect_largest_face(detector, image)

                if bbox is None:
                    stats["no_face"] += 1
                    if args.drop_no_face:
                        stats["dropped_no_face"] += 1
                        continue
                    cropped = image
                else:
                    stats["face_detected"] += 1
                    x, y, bw, bh = bbox
                    x1, y1, x2, y2 = expand_box(x, y, bw, bh, w, h, args.margin)
                    cropped = image[y1:y2, x1:x2]

                output_name = f"{class_name}_{out_index:04d}.jpg"
                save_jpeg(cropped, class_out / output_name)
                out_index += 1
                stats["saved"] += 1
            except Exception:
                stats["errors"] += 1

        summary["classes"][class_name] = stats
        print(
            f"{class_name}: input={stats['input']}, saved={stats['saved']}, "
            f"faces={stats['face_detected']}, no_face={stats['no_face']}, errors={stats['errors']}"
        )

    with (report_root / "face_extraction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Face extraction complete")
    print("[OK] Faces written to", output_root)
    print("[OK] Summary written to reports/face_extraction_summary.json")


if __name__ == "__main__":
    main()
