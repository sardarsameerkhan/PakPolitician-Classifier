"""
Quick training test - runs 2 epochs to verify pipeline works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import get_dataloaders
from train_models import ModelTrainer, create_resnet50
import torch


def quick_train_test():
    """Run quick training test"""
    print("\n" + "="*60)
    print("Quick Training Test (2 epochs)")
    print("="*60 + "\n")
    
    try:
        # Load data
        print("Loading dataset...")
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            root_dir="dataset",
            batch_size=16,
            num_workers=0,  # Use 0 for testing
            image_size=224,
        )
        
        num_classes = len(class_names)
        print(f"[OK] Dataset loaded with {num_classes} classes\n")
        
        # Create model
        print("Creating ResNet-50 model...")
        model = create_resnet50(num_classes)
        print("[OK] Model created\n")
        
        # Train
        print("Starting training...")
        trainer = ModelTrainer(model, "ResNet-50", num_classes)
        
        result = trainer.train(
            train_loader,
            val_loader,
            epochs=2,
            model_dir="models_test",
        )
        
        print(f"\nBest Val Acc: {result['best_val_acc']:.2f}%")
        print(f"Training Time: {result['elapsed_time']/60:.1f} minutes")
        
        print("\n" + "="*60)
        print("[SUCCESS] Training test passed!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_train_test()
    sys.exit(0 if success else 1)
