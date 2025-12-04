"""
Dataset loading and preprocessing utilities.
"""

import os

# Set environment variables BEFORE importing heavy libraries (for HPC compatibility)
# This prevents OpenBLAS threading issues on login nodes
if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'
if 'NUMEXPR_NUM_THREADS' not in os.environ:
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from collections import Counter

from .augmentation import get_train_transform, get_val_transform


# Emotion to emoji mapping
EMOTION_TO_EMOJI = {
    0: '😠',  # Angry
    1: '🤢',  # Disgust
    2: '😱',  # Fear
    3: '😄',  # Happy
    4: '😢',  # Sad
    5: '😲',  # Surprise
    6: '😐'   # Neutral
}

EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset loader.
    
    Args:
        data_dir: Path to dataset directory (should contain train/, test/, val/ subdirectories)
        split: One of 'train', 'test', or 'val'
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load images and labels
        self.images = []
        self.labels = []
        
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist!")
        
        # Iterate through emotion class folders
        for class_idx, emotion_name in enumerate(EMOTION_NAMES):
            class_dir = split_dir / emotion_name.lower()
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
                
            # Load all images in this class (support both .png and .jpg)
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_path in class_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {split} split")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at index.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image (grayscale)
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        
        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return image, label


def compute_class_weights(data_dir: str, split: str = 'train') -> torch.Tensor:
    """
    Compute class weights for handling class imbalance.
    
    Args:
        data_dir: Path to dataset directory
        split: Split to compute weights from (usually 'train')
        
    Returns:
        Tensor of class weights
    """
    split_dir = Path(data_dir) / split
    class_counts = []
    
    for emotion_name in EMOTION_NAMES:
        class_dir = split_dir / emotion_name.lower()
        if class_dir.exists():
            # Count both .png and .jpg files
            count = len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.jpeg')))
        else:
            count = 0
        class_counts.append(count)
    
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    
    # Compute weights: inverse frequency, normalized
    class_weights = 1.0 / (class_counts + 1e-6)  # Add small epsilon to avoid division by zero
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print(f"Class counts: {dict(zip(EMOTION_NAMES, class_counts.tolist()))}")
    print(f"Class weights: {dict(zip(EMOTION_NAMES, class_weights.tolist()))}")
    
    return class_weights


def get_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Create datasets
    train_dataset = FER2013Dataset(data_dir, split='train', transform=train_transform)
    val_dataset = FER2013Dataset(data_dir, split='val', transform=val_transform)
    test_dataset = FER2013Dataset(data_dir, split='test', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

