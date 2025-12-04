"""
Data augmentation pipeline using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Callable


def get_train_transform() -> Callable:
    """
    Get training data augmentation pipeline.
    
    Returns:
        Albumentations transform pipeline for training data.
    """
    return A.Compose([
        # Geometric augmentations (using Affine instead of ShiftScaleRotate)
        A.Affine(
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            p=0.7
        ),
        A.HorizontalFlip(p=0.5),
        
        # Occlusion simulation
        A.CoarseDropout(
            holes=2,
            height=8,
            width=8,
            p=0.3
        ),
        
        # Lighting variations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        
        # Normalize to [-1, 1] range
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])


def get_val_transform() -> Callable:
    """
    Get validation/test data transform pipeline (no augmentation).
    
    Returns:
        Albumentations transform pipeline for validation/test data.
    """
    return A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

