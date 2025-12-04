"""
Attention-Enhanced CNN with channel and spatial attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine
        out = avg_out + max_out
        out = out.view(b, c, 1, 1)
        
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate avg and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        out = self.conv(x_cat)
        out = self.sigmoid(out)
        
        return x * out


class AttentionBlock(nn.Module):
    """Combined channel and spatial attention block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionCNN(nn.Module):
    """
    Attention-Enhanced CNN architecture.
    
    Architecture:
    - 3 convolutional stages with attention blocks
    - Channel and spatial attention to focus on important facial regions
    """
    
    def __init__(self, num_classes: int = 7):
        super(AttentionCNN, self).__init__()
        
        # Stage 1: Feature Extraction
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 48x48 -> 24x24
        )
        self.attention1 = AttentionBlock(64)
        
        # Stage 2: Feature Refinement
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 24x24 -> 12x12
        )
        self.attention2 = AttentionBlock(128)
        
        # Stage 3: High-Level Features
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 12x12 -> 1x1
        )
        self.attention3 = AttentionBlock(256)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x = self.stage1(x)
        x = self.attention1(x)
        
        # Stage 2
        x = self.stage2(x)
        x = self.attention2(x)
        
        # Stage 3
        x = self.stage3(x)
        x = self.attention3(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

