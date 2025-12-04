"""
Evaluation metrics and confusion matrix utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from pathlib import Path
from typing import Tuple, Dict

from .dataset import EMOTION_NAMES, EMOTION_TO_EMOJI


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        verbose: Whether to print metrics
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    if verbose:
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    model_name: str = "Model"
):
    """
    Plot and save confusion matrix with emoji labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        model_name: Name of the model for title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    # Create labels with emojis
    labels = [f"{EMOTION_TO_EMOJI[i]}\n{name}" for i, name in enumerate(EMOTION_NAMES)]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Normalized Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {save_path}")


def get_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Get per-class metrics.
    
    Returns:
        Dictionary mapping class names to their metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class = {}
    for i, name in enumerate(EMOTION_NAMES):
        per_class[name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    return per_class

