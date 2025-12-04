"""
Evaluation script for all trained models.
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
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model, MODEL_REGISTRY
from utils.dataset import get_data_loaders, EMOTION_NAMES, EMOTION_TO_EMOJI
from utils.metrics import calculate_metrics, plot_confusion_matrix, get_per_class_metrics
from utils.visualization import visualize_attention_maps, plot_model_comparison


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    test_loader,
    device: torch.device,
    results_dir: Path
):
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = get_model(model_name, num_classes=7)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get model info
    total_params = checkpoint.get('total_params', sum(p.numel() for p in model.parameters()))
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(images))  # Per image
            
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_labels, all_preds, verbose=True)
    
    # Per-class metrics
    per_class_metrics = get_per_class_metrics(all_labels, all_preds)
    print("\nPer-Class Metrics:")
    print("-" * 60)
    for emotion, m in per_class_metrics.items():
        print(f"{emotion:12s}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, "
              f"F1={m['f1_score']:.3f}, Support={int(m['support'])}")
    
    # Average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    print(f"\nAverage inference time: {avg_inference_time:.2f} ms")
    
    # Save confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = results_dir / 'confusion_matrices' / f'{model_name}_confusion.png'
    plot_confusion_matrix(all_labels, all_preds, cm_path, model_name)
    
    # Generate attention maps (sample)
    print("\nGenerating attention maps (sample)...")
    try:
        # Get a sample batch
        sample_images, sample_labels = next(iter(test_loader))
        sample_images = sample_images[:10].to(device)
        sample_labels = sample_labels[:10]
        
        with torch.no_grad():
            sample_outputs = model(sample_images)
            sample_preds = sample_outputs.argmax(dim=1)
        
        # Try to find a suitable layer for Grad-CAM
        target_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is not None:
            attention_dir = results_dir / 'attention_maps' / model_name
            visualize_attention_maps(
                model,
                sample_images,
                sample_labels,
                sample_preds,
                attention_dir,
                model_name,
                target_layer=target_layer,
                num_samples=10
            )
        else:
            print("⚠️  Could not find suitable layer for attention visualization")
    except Exception as e:
        print(f"⚠️  Attention map generation failed: {e}")
    
    # Return results
    results = {
        'model_name': model_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'parameters': total_params,
        'inference_time_ms': avg_inference_time,
        'per_class': per_class_metrics
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Path to checkpoint directory (default: checkpoints)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to dataset directory (default: data)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Path to results directory (default: results)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all'] + list(MODEL_REGISTRY.keys()),
        help='Model to evaluate (default: all)'
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
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / args.checkpoint_dir
    data_dir = project_root / args.data_dir
    results_dir = project_root / args.results_dir
    
    # Create results directories
    (results_dir / 'confusion_matrices').mkdir(parents=True, exist_ok=True)
    (results_dir / 'attention_maps').mkdir(parents=True, exist_ok=True)
    (results_dir / 'comparison_plots').mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_data_loaders(
        str(data_dir),
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )
    
    # Find available checkpoints
    if args.model == 'all':
        available_models = []
        for model_name in MODEL_REGISTRY.keys():
            checkpoint_path = checkpoint_dir / f'{model_name}_best.pth'
            if checkpoint_path.exists():
                available_models.append(model_name)
    else:
        checkpoint_path = checkpoint_dir / f'{args.model}_best.pth'
        if checkpoint_path.exists():
            available_models = [args.model]
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
    
    if not available_models:
        print("❌ No checkpoints found!")
        return
    
    print(f"\nFound {len(available_models)} model(s) to evaluate:")
    for m in available_models:
        print(f"  - {m}")
    
    # Evaluate all models
    all_results = []
    comparison_data = {}
    
    for model_name in available_models:
        checkpoint_path = checkpoint_dir / f'{model_name}_best.pth'
        
        try:
            results = evaluate_model(
                model_name,
                checkpoint_path,
                test_loader,
                device,
                results_dir
            )
            all_results.append(results)
            comparison_data[model_name] = {
                'accuracy': results['accuracy'],
                'parameters': results['parameters'] / 1e6,  # Convert to millions
                'inference_time_ms': results['inference_time_ms']
            }
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison table
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    df = df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 
             'parameters', 'inference_time_ms']]
    df['accuracy'] = df['accuracy'] * 100  # Convert to percentage
    df['precision'] = df['precision'] * 100
    df['recall'] = df['recall'] * 100
    df['f1_score'] = df['f1_score'] * 100
    df['parameters'] = df['parameters'] / 1e6  # Convert to millions
    
    df.columns = ['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 
                  'F1-Score (%)', 'Parameters (M)', 'Inference Time (ms)']
    
    print("\n" + df.to_string(index=False))
    
    # Save comparison table
    csv_path = results_dir / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Comparison table saved to {csv_path}")
    
    # Generate comparison plot
    if len(comparison_data) > 1:
        print("\nGenerating comparison plots...")
        from utils.visualization import plot_model_comparison
        plot_path = results_dir / 'comparison_plots' / 'model_comparison.png'
        plot_model_comparison(comparison_data, plot_path)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

