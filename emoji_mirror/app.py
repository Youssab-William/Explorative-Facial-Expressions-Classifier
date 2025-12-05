"""
Emoji Mirror - Real-time facial expression to emoji mapping using webcam.

This Streamlit app uses your trained emotion detection model to detect
facial expressions from webcam feed and display the corresponding emoji.
"""

# CRITICAL: Set temp directory BEFORE any other imports
# This must happen before torch/dill imports that use tempfile
import os
import sys
from pathlib import Path

# Set custom temporary directory - use project directory as fallback
project_temp = Path(__file__).parent.parent / '.tmp'
try:
    project_temp.mkdir(exist_ok=True, mode=0o755)
    # Set environment variables for temp directory
    os.environ['TMPDIR'] = str(project_temp)
    os.environ['TMP'] = str(project_temp)
    os.environ['TEMP'] = str(project_temp)
    # Also set for tempfile module
    import tempfile
    tempfile.tempdir = str(project_temp)
except (OSError, PermissionError):
    # If we can't create the directory, try to use it anyway if it exists
    if project_temp.exists():
        os.environ['TMPDIR'] = str(project_temp)
        os.environ['TMP'] = str(project_temp)
        os.environ['TEMP'] = str(project_temp)
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Add project root to path (for models and utils)
# __file__ is emoji_mirror/app.py, so we need to go up 2 levels to get project root
# Use resolve() to ensure we get absolute paths
_app_file = Path(__file__).resolve()
# app.py is in emoji_mirror/, so:
# parent = emoji_mirror/
# parent.parent = project root (Explorative-Facial-Expressions-Classifier)
project_root = _app_file.parent.parent
sys.path.insert(0, str(project_root))

# Import from src (relative to emoji_mirror directory)
emoji_mirror_dir = Path(__file__).parent
sys.path.insert(0, str(emoji_mirror_dir))

from src.model_loader import EmotionModel
from src.image_processor import ImageProcessor
from src.face_detector import FaceDetector


# Page configuration
st.set_page_config(
    page_title="Emoji Mirror 🎭",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .emoji-display {
        font-size: 8rem;
        text-align: center;
        margin: 2rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .confidence-display {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_emotion_model(model_name: str = 'vit_tiny'):
    """Load emotion detection model (cached)."""
    try:
        # Use absolute path to avoid any path resolution issues
        checkpoint_dir = (project_root / 'checkpoints').resolve()
        expected_path = checkpoint_dir / f'{model_name}_best.pth'
        
        # Debug output (will show in terminal, not in Streamlit UI)
        print(f"DEBUG: Project root: {project_root.resolve()}")
        print(f"DEBUG: Checkpoint dir: {checkpoint_dir}")
        print(f"DEBUG: Expected checkpoint: {expected_path}")
        print(f"DEBUG: Checkpoint exists: {expected_path.exists()}")
        
        if not expected_path.exists():
            # List available files for debugging
            if checkpoint_dir.exists():
                available = list(checkpoint_dir.glob("*.pth"))
                available_names = [f.name for f in available]
                print(f"DEBUG: Available checkpoints: {available_names}")
                st.warning(f"Checkpoint {model_name}_best.pth not found. Available: {', '.join(available_names)}")
        
        model = EmotionModel(model_name=model_name, checkpoint_dir=str(checkpoint_dir))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Make sure checkpoint exists at: {project_root / 'checkpoints' / f'{model_name}_best.pth'}")
        return None


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Emoji Mirror</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Facial Expression to Emoji Detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = ['vit_tiny', 'attention_cnn', 'baseline_cnn']
        selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            index=0,
            help="Choose which trained model to use"
        )
        
        # Face detection toggle
        use_face_detection = st.checkbox(
            "Use Face Detection",
            value=True,
            help="Crop to face region for better accuracy"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum confidence to display prediction"
        )
        
        # Display all probabilities
        show_all_probs = st.checkbox(
            "Show All Probabilities",
            value=False,
            help="Display probability for each emotion"
        )
    
    # Load model
    model = load_emotion_model(selected_model)
    
    if model is None:
        st.error("❌ Failed to load model. Please check that checkpoints are available.")
        st.stop()
    
    # Initialize processors
    image_processor = ImageProcessor()
    face_detector = FaceDetector() if use_face_detection else None
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Webcam Feed")
        
        # Webcam input
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            st.error("❌ Could not open webcam. Please check your camera permissions.")
            st.stop()
        
        # Create placeholder for video
        video_placeholder = st.empty()
        emoji_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            start_button = st.button("▶️ Start Detection", type="primary")
        with col_stop:
            stop_button = st.button("⏹️ Stop Detection")
        
        # Detection state
        if 'detecting' not in st.session_state:
            st.session_state.detecting = False
        
        if start_button:
            st.session_state.detecting = True
        
        if stop_button:
            st.session_state.detecting = False
        
        # Detection loop
        if st.session_state.detecting:
            frame_count = 0
            last_prediction = None
            last_confidence = 0.0
            
            while st.session_state.detecting:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                # Process every 5th frame for performance
                frame_count += 1
                if frame_count % 5 == 0:
                    # Extract face if detection enabled
                    if use_face_detection and face_detector and face_detector.available:
                        face_roi = face_detector.extract_face_roi(frame)
                        if face_roi is not None:
                            # Convert BGR to RGB for display
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            display_frame = face_detector.draw_face_box(display_frame)
                            
                            # Process face ROI
                            try:
                                image_tensor = image_processor.preprocess(face_roi)
                                result = model.predict(image_tensor)
                                
                                if result['confidence'] >= confidence_threshold:
                                    last_prediction = result
                                    last_confidence = result['confidence']
                            except Exception as e:
                                st.warning(f"Processing error: {e}")
                        else:
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        # Use full frame
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            image_tensor = image_processor.preprocess(frame)
                            result = model.predict(image_tensor)
                            
                            if result['confidence'] >= confidence_threshold:
                                last_prediction = result
                                last_confidence = result['confidence']
                        except Exception as e:
                            st.warning(f"Processing error: {e}")
                    
                    # Update display
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
                else:
                    # Just display frame without processing
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if use_face_detection and face_detector and face_detector.available:
                        display_frame = face_detector.draw_face_box(display_frame)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
                # Display emoji and info
                if last_prediction:
                    with emoji_placeholder.container():
                        st.markdown(f'<div class="emoji-display">{last_prediction["emoji"]}</div>', unsafe_allow_html=True)
                    
                    with info_placeholder.container():
                        st.markdown(f'<div class="confidence-display">**{last_prediction["emotion"]}** ({last_confidence*100:.1f}% confidence)</div>', unsafe_allow_html=True)
                        
                        if show_all_probs:
                            st.subheader("All Probabilities")
                            prob_data = {
                                emotion: f"{prob*100:.1f}%"
                                for emotion, prob in last_prediction['all_probs'].items()
                            }
                            st.json(prob_data)
                
                time.sleep(0.03)  # ~30 FPS
        
        else:
            # Show static frame when not detecting
            ret, frame = video_capture.read()
            if ret:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if use_face_detection and face_detector and face_detector.available:
                    display_frame = face_detector.draw_face_box(display_frame)
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            emoji_placeholder.empty()
            info_placeholder.info("👆 Click 'Start Detection' to begin!")
        
        video_capture.release()
    
    with col2:
        st.header("ℹ️ Information")
        
        st.markdown("""
        ### How it works:
        1. **Start Detection** - Begin webcam feed
        2. **Face Detection** - Automatically finds your face
        3. **Emotion Analysis** - Model analyzes facial expression
        4. **Emoji Display** - Shows corresponding emoji
        
        ### Model Info:
        - **Current Model:** {model_name}
        - **Accuracy:** ~70% (ViT-Tiny)
        - **Input Size:** 48×48 grayscale
        
        ### Tips:
        - Ensure good lighting
        - Face the camera directly
        - Make clear expressions
        - Adjust confidence threshold if needed
        """.format(model_name=selected_model))
        
        st.markdown("---")
        st.markdown("### 📊 Emotion Classes")
        emotion_info = {
            '😠': 'Angry',
            '🤢': 'Disgust',
            '😱': 'Fear',
            '😄': 'Happy',
            '😢': 'Sad',
            '😲': 'Surprise',
            '😐': 'Neutral'
        }
        
        for emoji, name in emotion_info.items():
            st.markdown(f"{emoji} **{name}**")


if __name__ == "__main__":
    main()

