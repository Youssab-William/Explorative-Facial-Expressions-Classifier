"""
Model loading utilities for emotion detection.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models import get_model


class EmotionModel:
    """Wrapper class for emotion detection model."""
    
    def __init__(self, model_name: str = 'vit_tiny', checkpoint_dir: str = None):
        """
        Initialize emotion detection model.
        
        Args:
            model_name: Name of the model to load
            checkpoint_dir: Directory containing checkpoints (default: project checkpoints/)
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set checkpoint directory - use absolute paths
        if checkpoint_dir is None:
            checkpoint_dir = (project_root / 'checkpoints').resolve()
        else:
            checkpoint_dir = Path(checkpoint_dir).resolve()
        
        self.checkpoint_path = (checkpoint_dir / f'{model_name}_best.pth').resolve()
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        # Debug: print paths for troubleshooting
        print(f"DEBUG: Looking for checkpoint at: {self.checkpoint_path}")
        print(f"DEBUG: Absolute path: {self.checkpoint_path.absolute()}")
        print(f"DEBUG: Checkpoint dir exists: {self.checkpoint_path.parent.exists()}")
        print(f"DEBUG: Checkpoint dir: {self.checkpoint_path.parent}")
        
        if not self.checkpoint_path.exists():
            # List available files in checkpoint directory for debugging
            if self.checkpoint_path.parent.exists():
                available = list(self.checkpoint_path.parent.glob("*.pth"))
                available_str = ", ".join([f.name for f in available]) if available else "none"
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}\n"
                    f"Please train the model first or check the checkpoint directory.\n"
                    f"Available checkpoints: {available_str}"
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {self.checkpoint_path.parent}\n"
                    f"Please train the model first or check the checkpoint directory."
                )
        
        print(f"Loading {self.model_name} from {self.checkpoint_path}...")
        
        # Create model architecture
        self.model = get_model(self.model_name, num_classes=7)
        
        # Load weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def predict(self, image_tensor: torch.Tensor) -> dict:
        """
        Predict emotion from preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor [1, 1, 48, 48]
            
        Returns:
            Dictionary with prediction results:
            {
                'emotion': str,
                'emoji': str,
                'confidence': float,
                'all_probs': dict
            }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Emotion mappings
        EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        EMOTION_TO_EMOJI = {
            0: '😠',  # Angry
            1: '🤢',  # Disgust
            2: '😱',  # Fear
            3: '😄',  # Happy
            4: '😢',  # Sad
            5: '😲',  # Surprise
            6: '😐'   # Neutral
        }
        
        # Ensure tensor is on correct device
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # Get all probabilities
        all_probs = {
            EMOTION_NAMES[i]: probs[0][i].item()
            for i in range(len(EMOTION_NAMES))
        }
        
        return {
            'emotion': EMOTION_NAMES[pred_class],
            'emoji': EMOTION_TO_EMOJI[pred_class],
            'confidence': confidence,
            'all_probs': all_probs
        }

