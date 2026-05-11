"""
PyTorch Dataset and DataLoader for politician face images
Handles loading images from train/val/test folders with augmentation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from augmentation import get_train_augmentation, get_val_augmentation, get_test_augmentation


class PoliticianFaceDataset(Dataset):
    """Dataset class for politician face images"""
    
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: int = 224,
    ):
        """
        Args:
            root_dir: Path to dataset root (contains train/val/test folders)
            split: 'train', 'val', or 'test'
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        # Get augmentation transform
        if split == "train":
            self.transform = get_train_augmentation(image_size)
        elif split == "val":
            self.transform = get_val_augmentation(image_size)
        else:  # test
            self.transform = get_test_augmentation(image_size)
        
        # Build image list and labels
        self.images = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        
        self._load_images()
    
    def _load_images(self) -> None:
        """Load image paths and labels from split folder"""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Get all class folders
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = idx
            
            # Get all images in this class
            for img_file in sorted(class_dir.glob("*.jpg")):
                self.images.append(str(img_file))
                self.labels.append(idx)
        
        print(f"Loaded {len(self.images)} images from {self.split} split")
        print(f"Classes: {len(self.class_names)}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get single image and label"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": img_path,
            "class_name": self.class_names[label],
        }


def get_dataloaders(
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, val, and test sets
    
    Args:
        root_dir: Path to dataset root
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Image size
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = PoliticianFaceDataset(root_dir, split="train", image_size=image_size)
    val_dataset = PoliticianFaceDataset(root_dir, split="val", image_size=image_size)
    test_dataset = PoliticianFaceDataset(root_dir, split="test", image_size=image_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names


if __name__ == "__main__":
    print("✓ Dataset module ready")
