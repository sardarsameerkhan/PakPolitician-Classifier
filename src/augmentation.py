"""
Data Augmentation Pipeline for Training Images
Applies transformations: rotation, flipping, brightness, zooming, cropping
Only used on training data to prevent data leakage
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_augmentation(image_size: int = 224) -> A.Compose:
    """Get augmentation transforms for training data only"""
    return A.Compose(
        [
            # RandomResizedCrop combines resizing and cropping safely
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=0.5),
            
            # First resize to target size (for images not affected by RandomResizedCrop)
            A.Resize(height=image_size, width=image_size, always_apply=True),
            
            # Geometric transformations
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            
            # Brightness and contrast variations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            
            # Normalize and convert to tensor
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],   # ImageNet stds
                always_apply=True
            ),
            ToTensorV2(),
        ],
        bbox_params=None,
    )


def get_val_augmentation(image_size: int = 224) -> A.Compose:
    """Get augmentation transforms for validation/test data (no augmentation)"""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, always_apply=True),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True
            ),
            ToTensorV2(),
        ]
    )


def get_test_augmentation(image_size: int = 224) -> A.Compose:
    """Get augmentation transforms for test data (same as validation)"""
    return get_val_augmentation(image_size)


if __name__ == "__main__":
    print("✓ Augmentation pipeline ready")
    print("  Train: 7 augmentation techniques (rotation, flip, brightness, zoom, crop, noise, blur)")
    print("  Val/Test: Normalization only (no augmentation)")
