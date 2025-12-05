"""
Model architectures for facial expression classification.
"""

from .baseline_cnn import BaselineCNN
from .attention_cnn import AttentionCNN
from .vit_tiny import ViTTiny

__all__ = [
    'BaselineCNN',
    'AttentionCNN',
    'ViTTiny',
]

# Model name mapping
MODEL_REGISTRY = {
    'baseline_cnn': BaselineCNN,
    'attention_cnn': AttentionCNN,
    'vit_tiny': ViTTiny,
}


def get_model(model_name: str, num_classes: int = 7, **kwargs):
    """
    Get model by name.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](num_classes=num_classes, **kwargs)

