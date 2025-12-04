"""
Face detection utilities for webcam input.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class FaceDetector:
    """Face detection using OpenCV Haar Cascades."""
    
    def __init__(self):
        """Initialize face detector."""
        try:
            # Try to load Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise FileNotFoundError("Haar cascade file not found")
            
            self.available = True
        except Exception as e:
            print(f"⚠️ Face detection not available: {e}")
            print("   Continuing without face detection (will use full frame)")
            self.available = False
            self.face_cascade = None
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (x, y, w, h) if face detected, None otherwise
        """
        if not self.available:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        if len(faces) > 0:
            # Return largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            return tuple(largest_face)
        
        return None
    
    def extract_face_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face region of interest from frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Face ROI as numpy array, or None if no face detected
        """
        face_bbox = self.detect_face(frame)
        
        if face_bbox is None:
            return None
        
        x, y, w, h = face_bbox
        
        # Extract face region with some padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        
        return face_roi
    
    def draw_face_box(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw bounding box around detected face.
        
        Args:
            frame: Input frame
            color: BGR color for box
            
        Returns:
            Frame with bounding box drawn
        """
        face_bbox = self.detect_face(frame)
        
        if face_bbox is not None:
            x, y, w, h = face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        return frame

