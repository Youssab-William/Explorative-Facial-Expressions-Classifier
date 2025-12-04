"""
Ensemble of Specialists - separate experts for eyes, mouth, and full face regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertNetwork(nn.Module):
    """Individual expert network for a specific facial region."""
    
    def __init__(self, in_channels: int = 1, out_features: int = 128):
        super(ExpertNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EnsembleSpecialists(nn.Module):
    """
    Ensemble of Specialists architecture.
    
    Architecture:
    - Expert 1: Eyes region (rows 0-15)
    - Expert 2: Mouth region (rows 32-47)
    - Expert 3: Full face (48×48)
    - Fusion module with learned attention weights
    """
    
    def __init__(self, num_classes: int = 7):
        super(EnsembleSpecialists, self).__init__()
        
        # Expert 1: Eyes region (top 16 rows)
        self.expert_eyes = ExpertNetwork(in_channels=1, out_features=128)
        
        # Expert 2: Mouth region (bottom 16 rows)
        self.expert_mouth = ExpertNetwork(in_channels=1, out_features=128)
        
        # Expert 3: Full face
        self.expert_full = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.expert_full_fc = nn.Linear(256, 256)
        
        # Attention weights for expert combination
        self.attention = nn.Sequential(
            nn.Linear(512, 3),  # 128 + 128 + 256 = 512
            nn.Softmax(dim=1)
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract regions
        # Eyes: rows 0-15 (top half)
        eyes_region = x[:, :, 0:16, :]  # [batch, 1, 16, 48]
        
        # Mouth: rows 32-47 (bottom 16 rows)
        mouth_region = x[:, :, 32:48, :]  # [batch, 1, 16, 48]
        
        # Full face: all 48×48
        full_face = x  # [batch, 1, 48, 48]
        
        # Expert predictions
        expert1_out = self.expert_eyes(eyes_region)  # [batch, 128]
        expert2_out = self.expert_mouth(mouth_region)  # [batch, 128]
        
        expert3_features = self.expert_full(full_face)  # [batch, 256, 1, 1]
        expert3_features = expert3_features.view(expert3_features.size(0), -1)
        expert3_out = self.expert_full_fc(expert3_features)  # [batch, 256]
        
        # Concatenate expert features
        expert_features = torch.cat([expert1_out, expert2_out, expert3_out], dim=1)  # [batch, 512]
        
        # Compute attention weights
        attention_weights = self.attention(expert_features)  # [batch, 3]
        
        # Weighted combination (optional, can also just use concatenation)
        # For simplicity, we'll use concatenation and let the fusion network learn
        
        # Final classification
        out = self.fusion(expert_features)
        
        return out

