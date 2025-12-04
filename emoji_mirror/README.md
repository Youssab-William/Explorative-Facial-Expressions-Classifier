# 🎭 Emoji Mirror - Real-Time Emotion Detection

A real-time webcam-based facial expression to emoji mapping interface using deep learning.

## 🚀 Quick Start

### Prerequisites
- Trained model checkpoint (default: `vit_tiny_best.pth`)
- Webcam access
- Python 3.8+

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Also need parent project dependencies
cd ..
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
emoji_mirror/
├── app.py                 # Main Streamlit application
├── src/
│   ├── model_loader.py    # Model loading utilities
│   ├── image_processor.py # Image preprocessing
│   └── face_detector.py   # Face detection (optional)
├── assets/                # Static assets (if needed)
├── config/                # Configuration files (if needed)
└── requirements.txt       # Python dependencies
```

## 🎯 Features

- ✅ Real-time webcam emotion detection
- ✅ Face detection and cropping (optional)
- ✅ Multiple model support
- ✅ Confidence threshold adjustment
- ✅ Probability visualization
- ✅ Clean, modern UI

## ⚙️ Configuration

### Model Selection
Choose from available trained models:
- `vit_tiny` (default, 69.85% accuracy) ⭐ Recommended
- `attention_cnn` (62.56% accuracy)
- `hybrid_transformer` (55.71% accuracy)
- `ensemble` (53.15% accuracy)
- `baseline_cnn` (52.03% accuracy)
- `multiscale` (50.25% accuracy)

### Settings
- **Face Detection**: Automatically crop to face region
- **Confidence Threshold**: Minimum confidence to display prediction
- **Show All Probabilities**: Display probability for each emotion

## 🔧 Technical Details

### Model Loading
- Loads trained PyTorch model from checkpoints
- Uses same preprocessing as training
- Optimized for CPU inference (~50-100ms per frame)

### Image Processing
- Converts to grayscale
- Resizes to 48×48 (FER2013 input size)
- Applies same normalization as training

### Performance
- **CPU**: ~50-100ms per prediction
- **Frame Rate**: ~10-20 FPS (processes every 5th frame)
- **Memory**: ~200-300MB RAM

## 🐛 Troubleshooting

### Webcam not working
- Check camera permissions
- Try different camera index in code
- Ensure no other app is using camera

### Model not loading
- Verify checkpoint exists: `../checkpoints/vit_tiny_best.pth`
- Check model name matches available checkpoints

### Slow performance
- Reduce frame processing frequency
- Disable face detection
- Use smaller model (baseline_cnn)

## 📝 Notes

- Best results with good lighting
- Face camera directly for best accuracy
- Model trained on FER2013 dataset (48×48 grayscale)
- Real-time performance depends on hardware

