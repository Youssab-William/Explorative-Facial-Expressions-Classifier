"""
Visualization utilities for attention maps and training curves.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

# Try to import cv2 for colormap, but use matplotlib as fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not available. Attention maps will be disabled.")


def visualize_attention_maps(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    save_dir: Path,
    model_name: str = "Model",
    target_layer: Optional[torch.nn.Module] = None,
    num_samples: int = 10
):
    """
    Generate Grad-CAM attention heatmaps for model predictions.
    
    Args:
        model: Trained model
        images: Input images tensor [batch, channels, H, W]
        labels: True labels
        predictions: Predicted labels
        save_dir: Directory to save visualizations
        model_name: Name of the model
        target_layer: Target layer for Grad-CAM (if None, will try to find automatically)
        num_samples: Number of samples to visualize
    """
    if not GRAD_CAM_AVAILABLE:
        print("⚠️  Grad-CAM not available. Skipping attention maps.")
        return
    
    if target_layer is None:
        # Try to find a suitable layer (usually the last conv layer)
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            print("⚠️  Could not find suitable layer for Grad-CAM")
            return
    
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    
    # Visualize samples
    num_samples = min(num_samples, len(images))
    for i in range(num_samples):
        img = images[i:i+1]
        label = labels[i].item()
        pred = predictions[i].item()
        
        # Generate CAM
        targets = [ClassifierOutputTarget(pred)]
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert image to numpy
        img_np = img[0, 0].cpu().numpy()  # Grayscale
        img_np = (img_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        
        # Overlay CAM on image
        if CV2_AVAILABLE:
            # Use OpenCV colormap if available
            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
        else:
            # Use matplotlib colormap as fallback
            import matplotlib.cm as cm
            jet = cm.get_cmap('jet')
            heatmap = jet(grayscale_cam)[:, :, :3]  # Remove alpha channel
        
        cam_image = heatmap + np.float32(img_np[..., np.newaxis])
        cam_image = cam_image / np.max(cam_image)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        axes[2].imshow(cam_image)
        axes[2].set_title(f'Overlay (True: {label}, Pred: {pred})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{model_name}_attention_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Generated {num_samples} attention maps in {save_dir}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    model_name: str = "Model"
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
        model_name: Name of the model
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training and Validation Loss - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Training and Validation Accuracy - {model_name}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to {save_path}")


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Path
):
    """
    Plot comparison of all models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        save_path: Path to save the plot
    """
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    parameters = [results[m].get('parameters', 0) / 1e6 for m in models]  # Convert to millions
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax1.barh(models, accuracies, color='steelblue')
    ax1.set_xlabel('Test Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Parameter count comparison
    ax2.barh(models, parameters, color='coral')
    ax2.set_xlabel('Parameters (Millions)', fontsize=12)
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison plot saved to {save_path}")

