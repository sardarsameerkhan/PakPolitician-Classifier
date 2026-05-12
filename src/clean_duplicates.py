import json
from pathlib import Path
import shutil

report_path = Path("reports") / "dataset_check_report.json"
backup_root = Path("dataset_duplicates_backup")
backup_root.mkdir(exist_ok=True)

if not report_path.exists():
    print("Report not found:", report_path)
    raise SystemExit(1)

report = json.loads(report_path.read_text())
dups = report.get("duplicates", {})

removed = []

for md5, paths in dups.items():
    # Normalize paths
    paths = [Path(p) for p in paths]
    # Determine keep priority: any under dataset/train preferred, then val, then test
    keep = None
    for p in paths:
        if "\\train\\" in str(p) or "/train/" in str(p):
            keep = p
            break
    if keep is None:
        for p in paths:
            if "\\val\\" in str(p) or "/val/" in str(p):
                keep = p
                break
    if keep is None:
        keep = paths[0]

    for p in paths:
        if p == keep:
            continue
        if not p.exists():
            # maybe path uses backslashes vs forward slashes, try resolve relative to repo
            alt = Path(*p.parts)
            if alt.exists():
                p = alt
            else:
                removed.append({"path": str(p), "status": "missing"})
                continue
        # Move file to backup preserving original structure
        target = backup_root / p.parent
        target.mkdir(parents=True, exist_ok=True)
        dest = target / p.name
        try:
            shutil.move(str(p), str(dest))
            removed.append({"md5": md5, "moved": str(p), "to": str(dest)})
        except Exception as e:
            removed.append({"md5": md5, "path": str(p), "error": str(e)})

# Save report
out = Path("reports") / "duplicates_removed.json"
with open(out, "w") as f:
    json.dump(removed, f, indent=2)

print("Moved duplicate copies to", backup_root)
print("Saved reports/duplicates_removed.json")
