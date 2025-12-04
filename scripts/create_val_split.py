"""
Create validation split from training data.
FER2013 typically only has train/test, so we'll split train into train/val.
"""

import os
import sys

# Set environment variables BEFORE any imports
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import shutil
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import EMOTION_NAMES


def create_val_split(data_dir: Path, val_ratio: float = 0.125):
    """
    Create validation split from training data.
    
    Args:
        data_dir: Path to data directory
        val_ratio: Ratio of training data to use for validation (default: 12.5%)
    """
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    # Create val directory structure
    val_dir.mkdir(exist_ok=True)
    
    print(f"Creating validation split ({val_ratio*100:.1f}% of training data)...")
    print("=" * 60)
    
    total_moved = 0
    
    for emotion_name in EMOTION_NAMES:
        emotion_lower = emotion_name.lower()
        train_class_dir = train_dir / emotion_lower
        val_class_dir = val_dir / emotion_lower
        
        if not train_class_dir.exists():
            print(f"⚠️  Skipping {emotion_name}: directory not found")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(train_class_dir.glob(ext)))
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {emotion_name}")
            continue
        
        # Split into train and val (using random shuffle for reproducibility)
        random.seed(42)
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_ratio))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Create val class directory
        val_class_dir.mkdir(exist_ok=True)
        
        # Move validation files
        for img_file in val_files:
            dest = val_class_dir / img_file.name
            shutil.move(str(img_file), str(dest))
            total_moved += 1
        
        print(f"{emotion_name:12s}: {len(train_files):5d} train, {len(val_files):5d} val")
    
    print("=" * 60)
    print(f"✅ Created validation split: {total_moved} images moved")
    print(f"   Train: {train_dir}")
    print(f"   Val:   {val_dir}")
    
    return True


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Check if val already has images
    val_dir = data_dir / 'val'
    if val_dir.exists():
        val_images = sum(1 for _ in val_dir.rglob('*.png')) + sum(1 for _ in val_dir.rglob('*.jpg'))
        if val_images > 0:
            print(f"✅ Validation split already exists ({val_images} images)")
            response = input("Recreate? (y/n): ").strip().lower()
            if response != 'y':
                print("Keeping existing validation split.")
                return
    
    create_val_split(data_dir, val_ratio=0.125)


if __name__ == '__main__':
    main()

