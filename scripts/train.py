"""
Training script for all model architectures.
"""

import os

# Set environment variables BEFORE importing heavy libraries (for HPC compatibility)
if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'
if 'NUMEXPR_NUM_THREADS' not in os.environ:
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model, MODEL_REGISTRY
from utils.dataset import get_data_loaders, compute_class_weights
from utils.visualization import plot_training_curves


# Training configuration
BATCH_SIZE = 64
EPOCHS = 50
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0

# Model-specific configurations for fair comparison
MODEL_CONFIGS = {
    'baseline_cnn': {
        'lr': 0.001,
        'patience': 10,
    },
    'attention_cnn': {
        'lr': 0.001,
        'patience': 10,
    },
    'vit_tiny': {
        'lr': 0.00005,  # Very low for pretrained model
        'patience': 20,  # More patience
        'warmup_epochs': 5,  # Learning rate warmup
    },
}

def get_model_config(model_name: str):
    """Get configuration for specific model."""
    return MODEL_CONFIGS.get(model_name, {'lr': 0.001, 'patience': 10, 'warmup_epochs': 0})


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    model_name: str,
    data_dir: str,
    checkpoint_dir: Path,
    device: torch.device,
    resume: bool = False
):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'{model_name}_best.pth'
    history_path = checkpoint_dir / f'{model_name}_history.json'
    
    # Load data
    print("Loading data...")
    # Adjust num_workers and pin_memory based on device
    num_workers = 8 if device.type == 'cuda' else 2
    pin_memory = device.type == 'cuda'
    
    train_loader, val_loader, _ = get_data_loaders(
        data_dir,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Compute class weights
    class_weights = compute_class_weights(data_dir, split='train')
    class_weights = class_weights.to(device)
    
    # Create model
    print(f"Creating {model_name} model...")
    model = get_model(model_name, num_classes=7)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Get model-specific configuration
    config = get_model_config(model_name)
    learning_rate = config['lr']
    patience = config['patience']
    warmup_epochs = config.get('warmup_epochs', 0)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with optional warmup
    # Note: We'll handle warmup manually, then use ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Store initial learning rate for warmup
    initial_lr = learning_rate
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    
    if resume and checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['val_acc']
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {learning_rate}")
    print(f"Patience: {patience}")
    if warmup_epochs > 0:
        print(f"Warmup epochs: {warmup_epochs}")
    print()
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Handle learning rate warmup and scheduling
        old_lr = optimizer.param_groups[0]['lr']
        
        if epoch < warmup_epochs:
            # Warmup: linearly increase LR from 0.01 * initial_lr to initial_lr
            warmup_lr = initial_lr * (0.01 + 0.99 * (epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"  🔥 Warmup: LR = {warmup_lr:.6f} ({epoch+1}/{warmup_epochs})")
        else:
            # After warmup, use ReduceLROnPlateau
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_name': model_name,
                'total_params': total_params,
            }, checkpoint_path)
            
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping (using model-specific patience)
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_save_path = Path(__file__).parent.parent / 'results' / 'comparison_plots' / f'{model_name}_curves.png'
    plot_training_curves(history, plot_save_path, model_name)
    
    print(f"\n✅ Training complete for {model_name}!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Checkpoint saved: {checkpoint_path}")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train facial expression classification models')
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all'] + list(MODEL_REGISTRY.keys()),
        help='Model to train (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to dataset directory (default: data)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Path to checkpoint directory (default: checkpoints)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detects if not specified'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be very slow on CPU.")
        print("   To use GPU, submit a SLURM job: sbatch train_gpu.sh")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    checkpoint_dir = project_root / args.checkpoint_dir
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run: python scripts/setup_data.py")
        return
    
    # Determine models to train
    if args.model == 'all':
        models_to_train = list(MODEL_REGISTRY.keys())
    else:
        models_to_train = [args.model]
    
    # Train models
    results = {}
    for model_name in models_to_train:
        try:
            best_acc = train_model(
                model_name,
                str(data_dir),
                checkpoint_dir,
                device,
                resume=args.resume
            )
            results[model_name] = best_acc
            
            # Clear GPU cache between models
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for model_name, acc in results.items():
        print(f"{model_name:20s}: {acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()

