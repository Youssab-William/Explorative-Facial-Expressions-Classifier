"""
Hybrid CNN-Transformer architecture combining CNN feature extraction with Transformer global context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Residual block for CNN backbone."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0), :]
        return x


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture.
    
    Architecture:
    - CNN backbone extracts local features (edges, textures)
    - Features flattened into patches
    - Transformer encoder models global relationships between facial parts
    """
    
    def __init__(self, num_classes: int = 7, d_model: int = 512, nhead: int = 8, num_layers: int = 4):
        super(HybridCNNTransformer, self).__init__()
        
        # CNN Backbone
        self.cnn_backbone = nn.Sequential(
            # Initial conv + pooling
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 48x48 -> 24x24 -> 12x12
            
            # Residual blocks
            ResidualBlock(64, 128, stride=2),  # 12x12 -> 6x6
            ResidualBlock(128, 256, stride=2),  # 6x6 -> 3x3
        )
        
        # Patch projection: 256 channels * 3*3 = 9 patches
        self.patch_proj = nn.Linear(256, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=10)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.cnn_backbone(x)  # [batch, 256, 3, 3]
        
        # Flatten spatial dimensions and transpose for Transformer
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # [batch, 256, 9]
        x = x.permute(2, 0, 1)  # [9, batch, 256]
        
        # Project to d_model
        x = self.patch_proj(x)  # [9, batch, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # [9, batch, d_model]
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=0)  # [batch, d_model]
        
        # Classification
        out = self.classifier(x)
        
        return out

