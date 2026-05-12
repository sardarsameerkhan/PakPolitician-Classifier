import sys
import os
from pathlib import Path
import json

# Ensure src is on path so imports work like when running src scripts
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import torch

from dataset import get_dataloaders
from train_models import create_resnet50
from evaluate_models import ModelEvaluator

checkpoint = Path("models") / "ResNet-50_best.pth"
if not checkpoint.exists():
    print("Checkpoint not found:", checkpoint)
    raise SystemExit(1)

print("Loading dataloaders (this may take a moment)...")
train_loader, val_loader, test_loader, class_names = get_dataloaders(
    Path("dataset"), batch_size=32, num_workers=0, image_size=224
)

print(f"Classes: {len(class_names)}")

model = create_resnet50(len(class_names))
print("Loading checkpoint to CPU...")
state = torch.load(str(checkpoint), map_location="cpu")
model.load_state_dict(state)

# Evaluate on CPU to avoid interfering with GPU training
device = "cpu"
evaluator = ModelEvaluator(model, "ResNet-50", class_names, device=device)

results = evaluator.evaluate(test_loader)
acc = results["accuracy"] * 100
print(f"ResNet-50 Quick Eval Accuracy: {acc:.2f}%")

# Save a quick report
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)
with open(report_dir / "resnet_quick_eval.json", "w") as f:
    json.dump({"accuracy": results["accuracy"], "precision": results["precision"]}, f, indent=2)

print("Saved reports/resnet_quick_eval.json")
