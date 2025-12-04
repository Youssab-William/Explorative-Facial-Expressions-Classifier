"""
Setup script for downloading and preparing FER2013 dataset.
"""

import os
import sys

# Set environment variables BEFORE importing heavy libraries (for HPC compatibility)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import zipfile
import subprocess
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utils only when needed (after env vars are set)
# Note: These imports require opencv-python-headless (not opencv-python) for HPC compatibility
# OpenCV is only needed for Albumentations (data augmentation library)
try:
    from utils.dataset import get_data_loaders, compute_class_weights, EMOTION_NAMES
except ImportError as e:
    error_str = str(e)
    if 'libGL.so' in error_str or 'cv2' in error_str or 'opencv' in error_str.lower():
        print("\n❌ OpenCV import error detected!")
        print("OpenCV is required for Albumentations (data augmentation library).")
        print("\nSolution: Install opencv-python-headless (no GUI dependencies):")
        print("  pip uninstall opencv-python  # if installed")
        print("  pip install opencv-python-headless>=4.8.0")
        print("\nOr reinstall all requirements:")
        print("  pip install -r requirements.txt")
        print("\nNote: opencv-python-headless works on HPC systems without GUI libraries.")
        sys.exit(1)
    else:
        raise


def check_kaggle_api():
    """Check if Kaggle API is configured."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("❌ Kaggle API not configured!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' → downloads kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle API configured")
    return True


def download_dataset(data_dir: Path):
    """Download FER2013 dataset from Kaggle."""
    print("\n📥 Downloading FER2013 dataset from Kaggle...")
    
    try:
        # Download using Kaggle CLI
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'msambare/fer2013'],
            check=True,
            cwd=str(data_dir.parent)
        )
        print("✅ Download complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        return False


def extract_dataset(data_dir: Path):
    """Extract downloaded dataset."""
    zip_path = data_dir.parent / 'fer2013.zip'
    
    if not zip_path.exists():
        print(f"❌ Dataset zip file not found at {zip_path}")
        return False
    
    print("\n📦 Extracting dataset...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir.parent)
        print("✅ Extraction complete")
        
        # Remove zip file to save space
        zip_path.unlink()
        print("✅ Cleaned up zip file")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_dataset_structure(data_dir: Path):
    """Verify dataset has correct structure."""
    print("\n🔍 Verifying dataset structure...")
    
    required_splits = ['train', 'test', 'val']
    required_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    for split in required_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"⚠️  Warning: {split} directory not found")
            continue
        
        print(f"\n{split.upper()} split:")
        for class_name in required_classes:
            class_dir = split_dir / class_name
            if class_dir.exists():
                # Count both .png and .jpg files
                count = len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.jpeg')))
                print(f"  {class_name}: {count} images")
            else:
                print(f"  {class_name}: ❌ not found")
    
    return True


def explore_dataset(data_dir: Path):
    """Explore dataset statistics."""
    print("\n📊 Dataset Statistics:")
    print("=" * 50)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        str(data_dir),
        batch_size=64,
        num_workers=2
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Compute class weights
    print("\n" + "=" * 50)
    print("Class Weights (for handling imbalance):")
    class_weights = compute_class_weights(str(data_dir), split='train')
    
    return class_weights


def main():
    """Main setup function."""
    print("🚀 FER2013 Dataset Setup")
    print("=" * 50)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    train_dir = data_dir / 'train'
    if train_dir.exists() and any(train_dir.iterdir()):
        print("\n✅ Dataset already exists!")
        # Check if we have images
        image_count = sum(1 for _ in train_dir.rglob('*.png')) + sum(1 for _ in train_dir.rglob('*.jpg'))
        if image_count > 0:
            print(f"Found {image_count} images in dataset.")
            print("Using existing dataset...")
            verify_dataset_structure(data_dir)
            explore_dataset(data_dir)
            return
        else:
            print("⚠️  Dataset folders exist but no images found. Proceeding with download...")
    
    # Check Kaggle API
    if not check_kaggle_api():
        print("\n⚠️  Please configure Kaggle API first (see instructions above)")
        return
    
    # Download dataset
    if not download_dataset(data_dir):
        return
    
    # Extract dataset
    if not extract_dataset(data_dir):
        return
    
    # Verify structure
    verify_dataset_structure(data_dir)
    
    # Explore dataset
    explore_dataset(data_dir)
    
    print("\n✅ Dataset setup complete!")
    print(f"\nDataset location: {data_dir}")


if __name__ == '__main__':
    main()

