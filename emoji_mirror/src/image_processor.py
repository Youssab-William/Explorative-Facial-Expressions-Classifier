"""
Image preprocessing utilities for emotion detection.
"""

import numpy as np
from PIL import Image
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.augmentation import get_val_transform


class ImageProcessor:
    """Image preprocessing for emotion detection."""
    
    def __init__(self):
        """Initialize image processor with validation transforms."""
        self.transform = get_val_transform()
        self.target_size = (48, 48)  # FER2013 input size
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array (can be RGB, RGBA, or grayscale)
            
        Returns:
            Preprocessed tensor [1, 1, 48, 48]
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA
                pil_image = Image.fromarray(image, mode='RGBA')
                pil_image = pil_image.convert('L')  # Convert to grayscale
            elif image.shape[2] == 3:
                # RGB
                pil_image = Image.fromarray(image, mode='RGB')
                pil_image = pil_image.convert('L')  # Convert to grayscale
            else:
                # Already grayscale
                pil_image = Image.fromarray(image, mode='L')
        else:
            # Already grayscale
            pil_image = Image.fromarray(image, mode='L')
        
        # Resize to target size
        pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convert back to numpy
        image_array = np.array(pil_image)
        
        # Apply transform (same as validation during training)
        transformed = self.transform(image=image_array)
        image_tensor = transformed['image']  # [48, 48]
        
        # Add batch and channel dimensions
        image_tensor = image_tensor.unsqueeze(0)  # [1, 48, 48]
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 48, 48]
        
        return image_tensor
    
    def preprocess_from_file(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor [1, 1, 48, 48]
        """
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        return self.preprocess(image_array)

