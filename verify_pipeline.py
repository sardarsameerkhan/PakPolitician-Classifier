"""
Verification script to test dataset loading and augmentation pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import get_dataloaders
import torch


def verify_dataloaders():
    """Verify that dataloaders work correctly"""
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60 + "\n")
    
    try:
        # Load dataloaders
        print("Loading dataloaders...")
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            root_dir="dataset",
            batch_size=32,
            num_workers=2,
            image_size=224,
        )
        
        print("[OK] Dataloaders created successfully")
        print(f"  Classes: {len(class_names)}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}\n")
        
        # Get sample batch
        print("Loading sample batch from train set...")
        sample_batch = next(iter(train_loader))
        
        print(f"  Batch size: {sample_batch['image'].shape[0]}")
        print(f"  Image shape: {sample_batch['image'].shape}")
        print(f"  Label shape: {sample_batch['label'].shape}")
        print(f"  Labels: {sample_batch['label'].tolist()}")
        print(f"  Image dtype: {sample_batch['image'].dtype}")
        print(f"  Image min/max: {sample_batch['image'].min():.3f} / {sample_batch['image'].max():.3f}")
        print()
        
        # Check class names
        print(f"Classes: {class_names}\n")
        
        print("="*60)
        print("[SUCCESS] Dataset verification passed!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_dataloaders()
    sys.exit(0 if success else 1)
