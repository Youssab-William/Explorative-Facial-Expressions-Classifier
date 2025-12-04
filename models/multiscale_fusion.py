"""
Multi-Scale Feature Fusion Network - processes images at multiple resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Feature Fusion Network.
    
    Processes input at 3 resolutions:
    - Branch 1: 24×24 (coarse features - face shape)
    - Branch 2: 48×48 (normal features)
    - Branch 3: 96×96 (fine details - wrinkles, micro-expressions)
    """
    
    def __init__(self, num_classes: int = 7):
        super(MultiScaleFusion, self).__init__()
        
        # Branch 1: Coarse (24×24) - deeper network
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> 1x1
        )
        
        # Branch 2: Normal (48×48) - deeper network
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> 1x1
        )
        
        # Branch 3: Fine (96×96) - deeper network, but note: upsampling doesn't add info
        # Better approach: use attention to focus on important regions at original resolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 96x96 -> 48x48
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> 1x1
        )
        
        # Attention mechanism for branch fusion
        self.attention = nn.Sequential(
            nn.Linear(640, 128),  # 128 + 256 + 256 = 640
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # Fusion Network - enhanced
        # Branch1: 128, Branch2: 256, Branch3: 256 -> Total: 640
        self.fusion = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: Coarse (downsample to 24×24)
        x1 = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)
        x1 = self.branch1(x1)
        x1 = x1.view(x1.size(0), -1)  # [batch, 128]
        
        # Branch 2: Normal (keep 48×48)
        x2 = self.branch2(x)
        x2 = x2.view(x2.size(0), -1)  # [batch, 256]
        
        # Branch 3: Fine (upsample to 96×96) - note: upsampling doesn't add real info
        # But deeper network can still learn useful features
        x3 = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x3 = self.branch3(x3)
        x3 = x3.view(x3.size(0), -1)  # [batch, 256]
        
        # Concatenate features
        x_concat = torch.cat([x1, x2, x3], dim=1)  # [batch, 640]
        
        # Compute attention weights for branches
        attention_weights = self.attention(x_concat)  # [batch, 3]
        
        # Apply attention weights
        att_w1 = attention_weights[:, 0:1]  # [batch, 1]
        att_w2 = attention_weights[:, 1:2]  # [batch, 1]
        att_w3 = attention_weights[:, 2:3]  # [batch, 1]
        
        weighted_x1 = x1 * att_w1  # [batch, 128]
        weighted_x2 = x2 * att_w2  # [batch, 256]
        weighted_x3 = x3 * att_w3  # [batch, 256]
        
        # Concatenate weighted features
        x_fused = torch.cat([weighted_x1, weighted_x2, weighted_x3], dim=1)  # [batch, 640]
        
        # Final classification
        out = self.fusion(x_fused)
        
        return out

