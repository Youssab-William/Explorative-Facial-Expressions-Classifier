# 🎭 Facial Expression to Emoji Classifier

A comprehensive deep learning project that classifies facial expressions into 7 emotions and maps them to emojis. This project implements and compares 6 different neural network architectures, with a focus on custom hybrid designs.

## 📊 Project Overview

This project implements and evaluates 6 different architectures for facial expression recognition on the FER2013 dataset:

1. **Baseline CNN** - Simple 3-layer CNN (reference model)
2. **Attention-Enhanced CNN** - CNN with channel and spatial attention mechanisms
3. **Multi-Scale Feature Fusion** - Processes images at multiple resolutions
4. **Hybrid CNN-Transformer** - Combines CNN feature extraction with Transformer global context
5. **Ensemble of Specialists** - Separate experts for eyes, mouth, and full face regions
6. **ViT-Tiny** - Pretrained Vision Transformer (baseline)

## 🎯 Final Results

| Model | Validation Accuracy | Status | Notes |
|-------|-------------------|--------|-------|
| **ViT-Tiny** | **69.85%** | ⭐ Best | Excellent performance |
| **Attention-CNN** | 62.56% | ✅ Good | Consistent performer |
| **Hybrid Transformer** | 55.71% | ✅ Fixed | Improved from 25.11% |
| **Ensemble** | 53.15% | ✅ Stable | Stable performance |
| **Baseline CNN** | 52.03% | ✅ Normal | As expected |
| **Multi-Scale** | 50.25% | ✅ Improved | Improved from 37.28% |

### Key Achievements

- ✅ **ViT-Tiny achieved 69.85%** - Near state-of-the-art performance
- ✅ **Fixed training issues** - Hybrid Transformer improved from 25.11% to 55.71%
- ✅ **Architecture improvements** - Multi-Scale improved from 37.28% to 50.25%
- ✅ **Fair comparison** - All models properly trained with optimized hyperparameters

## 📦 Dataset

**FER2013** - 35,887 grayscale images (48×48 pixels) across 7 emotion classes:
- 😠 Angry
- 🤢 Disgust
- 😱 Fear
- 😄 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Setup Kaggle API (one-time)
# 1. Go to kaggle.com/account
# 2. Click "Create New API Token" → downloads kaggle.json
# 3. Place in ~/.kaggle/kaggle.json

# Download and prepare FER2013
python scripts/setup_data.py
python scripts/create_val_split.py
```

### 3. Train Models

```bash
# Train all models
python scripts/train.py --model all

# Or train specific model
python scripts/train.py --model vit_tiny
```

### 4. Evaluate Models

```bash
# Evaluate all trained models and generate visualizations
python scripts/evaluate.py
```

### 5. Run Emoji Mirror Interface

```bash
# Navigate to emoji mirror directory
cd emoji_mirror

# Install additional dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
Explorative-Facial-Expressions-Classifier/
├── data/                    # FER2013 dataset
│   ├── train/              # Training images
│   ├── test/               # Test images
│   └── val/                # Validation images
├── models/                  # Model architectures
│   ├── baseline_cnn.py
│   ├── attention_cnn.py
│   ├── multiscale_fusion.py
│   ├── hybrid_transformer.py
│   ├── ensemble.py
│   └── vit_tiny.py
├── utils/                   # Utility functions
│   ├── dataset.py          # Data loading
│   ├── augmentation.py     # Data augmentation
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py   # Plotting utilities
├── scripts/                 # Main execution scripts
│   ├── setup_data.py      # Dataset download/preparation
│   ├── create_val_split.py # Create validation split
│   ├── train.py            # Training script
│   └── evaluate.py        # Evaluation script
├── checkpoints/             # Saved model weights
├── results/                 # Evaluation outputs
│   ├── confusion_matrices/
│   ├── attention_maps/
│   └── comparison_plots/
├── demo/                    # Gradio demo interface
│   └── gradio_app.py
├── emoji_mirror/            # 🎭 Real-time emoji mirror interface
│   ├── app.py              # Streamlit main application
│   ├── src/
│   │   ├── model_loader.py  # Model loading utilities
│   │   ├── image_processor.py # Image preprocessing
│   │   └── face_detector.py   # Face detection
│   └── requirements.txt
├── requirements.txt
└── README.md
```

## 🎭 Emoji Mirror Interface

A real-time webcam-based interface that detects facial expressions and displays corresponding emojis.

### Features

- ✅ **Real-time webcam emotion detection**
- ✅ **Face detection and cropping** (optional, improves accuracy)
- ✅ **Multiple model support** (switch between trained models)
- ✅ **Confidence threshold adjustment**
- ✅ **Probability visualization** (see all emotion probabilities)
- ✅ **Clean, modern UI** with Streamlit

### How to Use

1. **Start the app:**
   ```bash
   cd emoji_mirror
   streamlit run app.py
   ```

2. **Configure settings:**
   - Select model (default: ViT-Tiny, 69.85% accuracy)
   - Enable/disable face detection
   - Adjust confidence threshold
   - Toggle probability display

3. **Use the interface:**
   - Click "Start Detection" to begin
   - Face the webcam
   - Make facial expressions
   - See emoji appear in real-time!

### Technical Details

**Model Loading:**
- Loads trained PyTorch model from checkpoints
- Uses same preprocessing as training
- Optimized for CPU inference (~50-100ms per frame)

**Image Processing:**
- Converts webcam feed to grayscale
- Resizes to 48×48 (FER2013 input size)
- Applies same normalization as training
- Optional face detection for better accuracy

**Performance:**
- **CPU**: ~50-100ms per prediction
- **Frame Rate**: ~10-20 FPS (processes every 5th frame)
- **Memory**: ~200-300MB RAM

### Architecture

The emoji mirror uses a clean, modular structure:

```
emoji_mirror/
├── app.py                 # Main Streamlit application
├── src/
│   ├── model_loader.py    # Loads and wraps trained models
│   ├── image_processor.py # Handles image preprocessing
│   └── face_detector.py   # OpenCV face detection (optional)
└── requirements.txt       # Dependencies
```

**Key Components:**

1. **EmotionModel** (`model_loader.py`):
   - Loads checkpoint
   - Wraps model for easy inference
   - Returns emotion, emoji, confidence, probabilities

2. **ImageProcessor** (`image_processor.py`):
   - Converts webcam frames to model input
   - Applies same transforms as training
   - Handles RGB → grayscale conversion

3. **FaceDetector** (`face_detector.py`):
   - Optional face detection using OpenCV
   - Crops to face region for better accuracy
   - Draws bounding box for visualization

## 🏗️ Model Architectures

### 1. Baseline CNN
- Simple 3-layer CNN
- Reference model for comparison
- 0.4M parameters

### 2. Attention-Enhanced CNN
- Channel attention (what to focus on)
- Spatial attention (where to focus)
- 1.2M parameters

### 3. Multi-Scale Feature Fusion
- Processes at 3 resolutions: 24×24, 48×48, 96×96
- Fuses multi-scale features
- 1.0M parameters

### 4. Hybrid CNN-Transformer
- CNN extracts local features
- Transformer models global relationships
- 14.1M parameters

### 5. Ensemble of Specialists
- Expert 1: Eyes region
- Expert 2: Mouth region
- Expert 3: Full face
- Attention-weighted fusion
- 0.8M parameters

### 6. ViT-Tiny
- Pretrained Vision Transformer
- Fine-tuned on FER2013
- 5.5M parameters

## 🔧 Training Configuration

Model-specific hyperparameters (optimized for fair comparison):

| Model | Learning Rate | Patience | Warmup | Notes |
|-------|--------------|----------|--------|-------|
| Baseline CNN | 0.001 | 10 | - | Standard |
| Attention-CNN | 0.001 | 10 | - | Standard |
| Multi-Scale | 0.001 | 10 | - | Standard |
| Hybrid Transformer | 0.0001 | 20 | 5 epochs | Lower LR for large model |
| Ensemble | 0.001 | 10 | - | Standard |
| ViT-Tiny | 0.00005 | 20 | 5 epochs | Very low LR for pretrained |

**Training Details:**
- Batch Size: 64
- Epochs: 50 (with early stopping)
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Loss: Weighted CrossEntropy (handles class imbalance)

## 📈 Training Improvements

### Issues Fixed

1. **Hybrid Transformer & ViT-Tiny Early Stopping:**
   - **Problem**: Models stopped at ~25% accuracy
   - **Solution**: Lower learning rates, warmup, more patience
   - **Result**: ViT-Tiny improved to 69.85%, Hybrid Transformer to 55.71%

2. **Multi-Scale Architecture:**
   - **Problem**: Poor performance (37.28%)
   - **Solution**: Deeper branches, attention mechanism, enhanced fusion
   - **Result**: Improved to 50.25%

3. **Ensemble Architecture:**
   - **Status**: Working correctly (53.15%)
   - **Note**: Could be improved further with better fusion

## 🎓 Key Insights

1. **ViT-Tiny performs best** (69.85%) - Pretrained transformers work well with proper fine-tuning
2. **Hyperparameter tuning critical** - Lower learning rates essential for large models
3. **Attention mechanisms help** - Attention-CNN performs well (62.56%)
4. **Architecture matters** - Multi-scale needs better design to reach potential
5. **Fair comparison achieved** - All models properly trained with optimized configs

## 🚀 Deployment

### Using Trained Models

The trained models can be used in two ways:

1. **Gradio Interface** (existing):
   ```bash
   python demo/gradio_app.py
   ```

2. **Emoji Mirror** (new, recommended):
   ```bash
   cd emoji_mirror
   streamlit run app.py
   ```

### Model Loading (Programmatic)

```python
from emoji_mirror.src.model_loader import EmotionModel
from emoji_mirror.src.image_processor import ImageProcessor

# Load model
model = EmotionModel(model_name='vit_tiny')

# Process image
processor = ImageProcessor()
image_tensor = processor.preprocess(image_array)

# Predict
result = model.predict(image_tensor)
print(f"Emotion: {result['emotion']} {result['emoji']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
```

## 📝 Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (for ViT)
- albumentations >= 1.3.0
- streamlit >= 1.28.0 (for emoji mirror)
- opencv-python-headless >= 4.8.0
- gradio >= 4.0.0 (for demo)

## 🔬 Evaluation

Run comprehensive evaluation:
```bash
python scripts/evaluate.py
```

Generates:
- Confusion matrices for each model
- Per-class metrics (precision, recall, F1)
- Comparison plots
- Attention visualizations

## 📊 Performance Benchmarks

**Inference Speed:**
- ViT-Tiny on CPU: ~50-100ms per image
- ViT-Tiny on GPU: ~5-10ms per image
- Suitable for real-time webcam (10-20 FPS)

**Model Sizes:**
- ViT-Tiny: 64MB
- Hybrid Transformer: 162MB
- Attention-CNN: 14MB
- Baseline CNN: 4.7MB

## 🎯 Future Improvements

Potential improvements for each model:
- **ViT-Tiny**: +2-3% (progressive resizing, better augmentation)
- **Attention-CNN**: +3-5% (deeper attention, multi-head)
- **Hybrid Transformer**: +5-10% (better backbone, more layers)
- **Ensemble**: +10-15% (better fusion, trained experts)
- **Multi-Scale**: +10-15% (FPN, cross-scale attention)

## 📚 Documentation

- **Training logs**: Check `train_*.out` files
- **Model checkpoints**: `checkpoints/*_best.pth`
- **Results**: `results/` directory
- **Emoji Mirror**: See `emoji_mirror/README.md`

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with architectures
- Improve hyperparameters
- Add new features to emoji mirror
- Optimize inference speed

## 📄 License

This project is for educational purposes.

## 👤 Author

University Deep Learning Course Project

---

**Note**: This project was developed and tested on HPC with NVIDIA A100 GPUs. For local development, ensure you have appropriate hardware or use CPU inference (slower but functional).
