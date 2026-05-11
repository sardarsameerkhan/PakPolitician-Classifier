"""
Model Evaluation Script
Generate evaluation metrics, confusion matrix, training curves, and misclassified samples
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from dataset import get_dataloaders


class ModelEvaluator:
    """Evaluator for face classification models"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        class_names: list[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)
    
    def evaluate(self, test_loader) -> dict[str, Any]:
        """Evaluate model on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []
        
        print(f"Evaluating {self.model_name}...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                paths = batch["path"]
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_paths.extend(paths)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
        )
        
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Get top 5 misclassified
        misclassified_indices = np.where(all_preds != all_labels)[0]
        misclassified_probs = np.max(all_probs[misclassified_indices], axis=1)
        top_misclassified_idx = np.argsort(-misclassified_probs)[:5]
        
        top_misclassified = []
        for idx in top_misclassified_idx:
            actual_idx = misclassified_indices[idx]
            top_misclassified.append({
                "path": all_paths[actual_idx],
                "true_class": self.class_names[all_labels[actual_idx]],
                "predicted_class": self.class_names[all_preds[actual_idx]],
                "confidence": float(np.max(all_probs[actual_idx])),
            })
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": conf_matrix.tolist(),
            "top_misclassified": top_misclassified,
            "all_preds": all_preds.tolist(),
            "all_labels": all_labels.tolist(),
            "all_probs": all_probs.tolist(),
        }
    
    def plot_confusion_matrix(self, evaluation_results: dict[str, Any]) -> None:
        """Plot and save confusion matrix heatmap"""
        conf_matrix = np.array(evaluation_results["confusion_matrix"])
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = Path("reports") / f"{self.model_name}_confusion_matrix.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to {output_path}")
        plt.close()
    
    def plot_misclassified(self, evaluation_results: dict[str, Any]) -> None:
        """Plot top 5 misclassified samples"""
        top_misclassified = evaluation_results["top_misclassified"]
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.suptitle(f"Top 5 Misclassified - {self.model_name}", fontsize=14)
        
        for idx, item in enumerate(top_misclassified):
            try:
                import cv2
                img = cv2.imread(item["path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img)
            except:
                axes[idx].text(0.5, 0.5, "Image not found", ha="center", va="center")
            
            axes[idx].set_title(
                f"True: {item['true_class']}\n"
                f"Pred: {item['predicted_class']}\n"
                f"Conf: {item['confidence']:.2f}",
                fontsize=9,
            )
            axes[idx].axis("off")
        
        output_path = Path("reports") / f"{self.model_name}_misclassified.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Misclassified samples saved to {output_path}")
        plt.close()


def plot_training_curves(history: dict[str, list], model_name: str) -> None:
    """Plot training vs validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(history["train_loss"], label="Train Loss", marker="o", markersize=4)
    ax1.plot(history["val_loss"], label="Val Loss", marker="s", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(history["train_acc"], label="Train Accuracy", marker="o", markersize=4)
    ax2.plot(history["val_acc"], label="Val Accuracy", marker="s", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name} - Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    output_path = Path("reports") / f"{model_name}_training_curves.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Training curves saved to {output_path}")
    plt.close()


def save_evaluation_report(
    model_name: str,
    evaluation_results: dict[str, Any],
    class_names: list[str],
) -> None:
    """Save evaluation results to JSON and text files"""
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    # Save JSON report
    json_path = report_dir / f"{model_name}_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"✓ Evaluation JSON saved to {json_path}")
    
    # Save text report
    text_path = report_dir / f"{model_name}_report.txt"
    with open(text_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy:  {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision: {evaluation_results['precision']:.4f}\n")
        f.write(f"  Recall:    {evaluation_results['recall']:.4f}\n")
        f.write(f"  F1-Score:  {evaluation_results['f1']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write(f"{'Class Name':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 66 + "\n")
        for class_name, metrics in evaluation_results["per_class_metrics"].items():
            f.write(
                f"{class_name:<30} {metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}\n"
            )
        
        f.write("\n" + "="*60 + "\n")
        f.write("Top 5 Misclassified Samples:\n")
        f.write("="*60 + "\n")
        for i, sample in enumerate(evaluation_results["top_misclassified"], 1):
            f.write(f"\n{i}. {sample['path']}\n")
            f.write(f"   True:      {sample['true_class']}\n")
            f.write(f"   Predicted: {sample['predicted_class']}\n")
            f.write(f"   Confidence: {sample['confidence']:.4f}\n")
    
    print(f"✓ Evaluation report saved to {text_path}")


def main():
    """Evaluate trained models"""
    
    # Load dataset
    dataset_root = Path("dataset")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        dataset_root,
        batch_size=32,
        num_workers=4,
        image_size=224,
    )
    
    # Load trained models
    models_to_evaluate = [
        ("ResNet-50", "models/ResNet-50_best.pth"),
        ("EfficientNet-B0", "models/EfficientNet-B0_best.pth"),
    ]
    
    results = {}
    
    for model_name, model_path in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Load model
        if model_name == "ResNet-50":
            from train_models import create_resnet50
            model = create_resnet50(len(class_names))
        else:
            from train_models import create_efficientnet_b0
            model = create_efficientnet_b0(len(class_names))
        
        model.load_state_dict(torch.load(model_path))
        
        # Evaluate
        evaluator = ModelEvaluator(model, model_name, class_names)
        evaluation_results = evaluator.evaluate(test_loader)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(evaluation_results)
        evaluator.plot_misclassified(evaluation_results)
        
        # Save report
        save_evaluation_report(model_name, evaluation_results, class_names)
        
        results[model_name] = evaluation_results
        
        print(f"\nAccuracy: {evaluation_results['accuracy']*100:.2f}%")
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
