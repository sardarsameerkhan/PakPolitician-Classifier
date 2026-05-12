"""
Model Training Script
Train multiple pretrained models (ResNet-50, EfficientNet) on politician face dataset
Includes MLflow tracking and model checkpointing
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from collections import defaultdict

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloaders


class ModelTrainer:
    """Trainer for face classification models"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.warmup_epochs = 3
        self.head_lr = 1e-3
        self.backbone_lr = 1e-4
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = self._build_optimizer()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.1, patience=3
        )
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _build_optimizer(self) -> optim.Optimizer:
        """Create a two-group optimizer: fast head warmup and slower backbone fine-tuning."""
        head_params = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc") or name.startswith("classifier"):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.backbone_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})

        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": self.head_lr}]

        return optim.Adam(param_groups, weight_decay=1e-4)

    def _set_training_stage(self, epoch: int) -> None:
        """Use head-only warmup first, then fine-tune the full network."""
        for name, param in self.model.named_parameters():
            if name.startswith("fc") or name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = epoch > self.warmup_epochs

        if epoch <= self.warmup_epochs:
            for group in self.optimizer.param_groups:
                group["lr"] = self.head_lr if group["params"] and group["params"][0].requires_grad else 0.0
        else:
            for group in self.optimizer.param_groups:
                if group["params"] and group["params"][0].requires_grad:
                    group["lr"] = self.backbone_lr if len(group["params"]) > 0 and not group["params"][0].shape == torch.Size([]) else self.head_lr

            # Give the classifier a slightly higher learning rate during fine-tuning.
            for group in self.optimizer.param_groups:
                if group["params"] and group["params"][0].requires_grad:
                    sample_name = None
                    for name, param in self.model.named_parameters():
                        if param is group["params"][0]:
                            sample_name = name
                            break
                    if sample_name and (sample_name.startswith("fc") or sample_name.startswith("classifier")):
                        group["lr"] = self.head_lr * 0.3
                    else:
                        group["lr"] = self.backbone_lr
    
    def train_epoch(self, train_loader) -> tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training {self.model_name}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                "loss": total_loss / (pbar.n + 1),
                "acc": 100 * correct / total,
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating {self.model_name}"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        model_dir: str | Path = "models",
    ) -> dict[str, Any]:
        """Train model for specified epochs"""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        best_val_acc = 0
        best_model_path = model_dir / f"{self.model_name}_best.pth"
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            self._set_training_stage(epoch)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # MLflow tracking (with error handling for Windows path encoding issues)
            try:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, step=epoch)
            except Exception as e:
                pass  # Silently skip MLflow if path encoding fails
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                print(f"[BEST] Model saved (Val Acc: {best_val_acc:.2f}%)")
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
        
        elapsed_time = time.time() - start_time
        print(f"\n[DONE] Training completed in {elapsed_time/60:.1f} minutes")
        
        return {
            "best_val_acc": best_val_acc,
            "best_model_path": str(best_model_path),
            "history": self.history,
            "elapsed_time": elapsed_time,
        }


def create_resnet50(num_classes: int) -> nn.Module:
    """Create ResNet-50 model"""
    from torchvision import models
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_efficientnet_b0(num_classes: int) -> nn.Module:
    """Create EfficientNet-B0 model"""
    from torchvision import models
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def main():
    """Train models"""
    
    # Configuration
    dataset_root = Path("dataset")
    batch_size = 32
    epochs = 40
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        dataset_root,
        batch_size=batch_size,
        num_workers=4,
        image_size=224,
    )
    num_classes = len(class_names)
    print(f"[OK] Classes: {num_classes}\n")
    
    # Train models
    models_to_train = [
        ("ResNet-50", create_resnet50(num_classes)),
        ("EfficientNet-B0", create_efficientnet_b0(num_classes)),
    ]
    
    results = {}
    
    for model_name, model in models_to_train:
        print(f"\nTraining {model_name}...")
        trainer = ModelTrainer(
            model,
            model_name,
            num_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        result = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            model_dir="models",
        )
        
        results[model_name] = result
    
    # Save results
    results_path = Path("reports") / "training_results.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump({
            k: {
                "best_val_acc": v["best_val_acc"],
                "best_model_path": v["best_model_path"],
                "elapsed_time": v["elapsed_time"],
            }
            for k, v in results.items()
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name}: Best Val Acc = {result['best_val_acc']:.2f}%")
    
    print(f"\n[OK] Results saved to {results_path}")


if __name__ == "__main__":
    main()

