"""
ViT-Tiny (Vision Transformer) using timm library.
"""

import torch
import torch.nn as nn
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. ViT-Tiny will not work.")


class ViTTiny(nn.Module):
    """
    ViT-Tiny architecture using pretrained weights from timm.
    
    Uses a pretrained Vision Transformer and fine-tunes for emotion classification.
    """
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(ViTTiny, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for ViT-Tiny. Install with: pip install timm")
        
        # Load pretrained ViT-Tiny
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            in_chans=1  # Grayscale input
        )
        
        # Get feature dimension
        feature_dim = self.vit.num_features  # Usually 192 for ViT-Tiny
        
        # Custom classifier for emotion classification
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT expects 224x224 input, but we have 48x48
        # Upsample to 224x224 for pretrained model
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        features = self.vit(x)
        
        # Classify
        out = self.classifier(features)
        
        return out

