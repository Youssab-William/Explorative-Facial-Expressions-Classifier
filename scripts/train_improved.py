"""
Improved training script with model-specific hyperparameters and warmup.
"""

import os

# Set environment variables BEFORE importing heavy libraries
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model, MODEL_REGISTRY
from utils.dataset import get_data_loaders, compute_class_weights
from utils.visualization import plot_training_curves


# Model-specific configurations
MODEL_CONFIGS = {
    'hybrid_transformer': {
        'lr': 0.0001,  # Lower LR for large model
        'warmup_epochs': 5,
        'patience': 15,
        'grad_accumulation': 2,
        'weight_decay': 0.0001,
    },
    'vit_tiny': {
        'lr': 0.00005,  # Very low for pretrained
        'warmup_epochs': 5,
        'patience': 15,
        'grad_accumulation': 1,
        'weight_decay': 0.0001,
        'freeze_backbone': True,  # Freeze early layers
    },
    'default': {
        'lr': 0.001,
        'warmup_epochs': 0,
        'patience': 10,
        'grad_accumulation': 1,
        'weight_decay': 0.0001,
    }
}

BATCH_SIZE = 64
EPOCHS = 50
GRAD_CLIP = 1.0


def get_model_config(model_name: str):
    """Get configuration for specific model."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['default'])


def train_epoch_with_accumulation(model, train_loader, criterion, optimizer, device, accumulation_steps=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
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


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create warmup + cosine annealing scheduler."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    # Warmup scheduler
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Cosine annealing after warmup
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # Combine
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )


def train_model_improved(
    model_name: str,
    data_dir: str,
    checkpoint_dir: Path,
    device: torch.device
):
    """Train model with improved hyperparameters."""
    config = get_model_config(model_name)
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Improved)")
    print(f"{'='*60}\n")
    print(f"Configuration:")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Warmup Epochs: {config['warmup_epochs']}")
    print(f"  Patience: {config['patience']}")
    print(f"  Gradient Accumulation: {config['grad_accumulation']}")
    print()
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'{model_name}_improved_best.pth'
    history_path = checkpoint_dir / f'{model_name}_improved_history.json'
    
    # Load data
    print("Loading data...")
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
    
    # Freeze backbone if needed (for ViT)
    if config.get('freeze_backbone', False) and model_name == 'vit_tiny':
        print("Freezing early transformer layers...")
        for name, param in model.vit.named_parameters():
            if 'blocks.0' in name or 'blocks.1' in name or 'blocks.2' in name:
                param.requires_grad = False
                print(f"  Frozen: {name}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    scheduler = get_warmup_scheduler(
        optimizer,
        config['warmup_epochs'],
        EPOCHS
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Effective batch size: {BATCH_SIZE * config['grad_accumulation']}")
    print()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=config['grad_accumulation']
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate (step per epoch)
        scheduler.step()
        
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
            print(f"  Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_save_path = Path(__file__).parent.parent / 'results' / 'comparison_plots' / f'{model_name}_improved_curves.png'
    plot_training_curves(history, plot_save_path, f'{model_name} (Improved)')
    
    print(f"\n✅ Training complete for {model_name}!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Checkpoint saved: {checkpoint_path}")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train models with improved hyperparameters')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help='Model to train'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Path to checkpoint directory'
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
    else:
        print("⚠️  WARNING: No GPU detected!")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    checkpoint_dir = project_root / args.checkpoint_dir
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Train model
    try:
        best_acc = train_model_improved(
            args.model,
            str(data_dir),
            checkpoint_dir,
            device
        )
        print(f"\n✅ Final accuracy: {best_acc:.2f}%")
    except Exception as e:
        print(f"❌ Error training {args.model}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

