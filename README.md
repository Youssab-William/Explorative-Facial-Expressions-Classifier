# Facial Expression to Emoji Classifier

This project implements and compares different neural network architectures for classifying facial expressions into 7 emotions using the FER2013 dataset. we wanted it be an explorative project so we worked with three models: a simple baseline CNN, an attention-enhanced CNN, and a Vision Transformer (ViT-Tiny).

## Project Overview

We tested three different architectures to see how they perform on facial expression recognition:

1. **Baseline CNN** - A simple 3-layer CNN to use as a reference
2. **Attention-Enhanced CNN** - Added channel and spatial attention mechanisms to see if they help
3. **ViT-Tiny** - Used a pretrained Vision Transformer and fine-tuned it on FER2013

## Results

Here's how each model performed:

| Model | Validation Accuracy | Notes |
|-------|-------------------|-------|
| **ViT-Tiny** | **69.85%** | Best performer - pretrained transformers work really well |
| **Attention-CNN** | 62.56% | Attention mechanisms definitely helped |
| **Baseline CNN** | 52.03% | Simple baseline, about what I expected |

The ViT-Tiny model did the best, which makes sense since it's pretrained on ImageNet. The attention mechanisms in the second model gave a nice boost over the baseline. Overall, we're happy with how the models turned out after tuning the hyperparameters.

## Dataset

I used the **FER2013** dataset which has 35,887 grayscale images (48×48 pixels) across 7 emotion classes:
- 😠 Angry
- 🤢 Disgust
- 😱 Fear
- 😄 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

## Getting Started

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

First, set up Kaggle API:
1. Go to kaggle.com/account
2. Click "Create New API Token" - this downloads `kaggle.json`
3. Place it in `~/.kaggle/kaggle.json`

Then download the dataset:

```bash
# Download FER2013 dataset
kaggle datasets download -d msambare/fer2013

# Extract it
unzip fer2013.zip -d data/

# Create validation split
python scripts/create_val_split.py
```

### 3. Train Models

```bash
# Train all models
python scripts/train.py --model all

# Or train a specific model
python scripts/train.py --model vit_tiny
```

### 4. Evaluate

```bash
# Evaluate all trained models and generate visualizations
python scripts/evaluate.py
```

This will generate confusion matrices, per-class metrics, and comparison plots in the `results/` directory.

## Project Structure

```
Explorative-Facial-Expressions-Classifier/
├── data/                    # FER2013 dataset
│   ├── train/              # Training images
│   ├── test/               # Test images
│   └── val/                # Validation images
├── models/                  # Model architectures
│   ├── baseline_cnn.py
│   ├── attention_cnn.py
│   └── vit_tiny.py
├── utils/                   # Utility functions
│   ├── dataset.py          # Data loading
│   ├── augmentation.py     # Data augmentation
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Plotting utilities
├── scripts/                 # Main execution scripts
│   ├── create_val_split.py # Create validation split
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── checkpoints/             # Saved model weights
├── results/                 # Evaluation outputs
│   ├── confusion_matrices/
│   ├── attention_maps/
│   └── comparison_plots/
├── demo/                    # Gradio demo interface
│   └── gradio_app.py
├── emoji_mirror/            # Streamlit interface (work in progress)
│   ├── app.py              # Streamlit main application
│   ├── src/
│   │   ├── model_loader.py  # Model loading utilities
│   │   ├── image_processor.py # Image preprocessing
│   │   └── face_detector.py   # Face detection
│   ├── run_app.sh          # Wrapper script for running
│   └── requirements.txt
├── requirements.txt
└── README.md
```

## Model Architectures

### Baseline CNN
A simple 3-layer CNN with about 0.4M parameters. I used this as a reference point to compare against the more complex models.

### Attention-Enhanced CNN
Added channel attention (to focus on important features) and spatial attention (to focus on important regions). This has about 1.2M parameters and performed noticeably better than the baseline.

### ViT-Tiny
Used a pretrained Vision Transformer from the `timm` library and fine-tuned it on FER2013. It has 5.5M parameters and ended up being the best performer. The pretrained weights really helped here.

## Training Details

I tuned the hyperparameters for each model to make sure the comparison was fair:

| Model | Learning Rate | Patience | Warmup |
|-------|--------------|----------|--------|
| Baseline CNN | 0.001 | 10 | - |
| Attention-CNN | 0.001 | 10 | - |
| ViT-Tiny | 0.00005 | 20 | 5 epochs |

The ViT-Tiny needed a much lower learning rate since it's pretrained. I also added a warmup period for it. All models used:
- Batch size: 64
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Loss: Weighted CrossEntropy (to handle class imbalance)
- Early stopping based on validation accuracy

## What we Learned

1. **Pretrained models are powerful** - The ViT-Tiny with ImageNet pretraining did significantly better than training from scratch
2. **Hyperparameter tuning matters** - The ViT needed a very low learning rate (0.00005) to fine-tune properly
3. **Attention helps** - The attention mechanisms gave a solid boost over the baseline
4. **Simple baselines are useful** - Having a baseline CNN helped me understand how much the other improvements actually mattered




## Emoji Mirror Interface (Work in Progress)

We started working on a real-time webcam interface using Streamlit in the `emoji_mirror/` directory. The idea was to have a live interface where you could see your facial expression detected in real-time with the corresponding emoji displayed. However, I ran into some issues with checkpoint loading and path resolution, so it's not fully working yet. The code is there if you want to take a look or help finish it, but it's not ready for use right now.


## Notes

- This project was developed and tested on HPC with NVIDIA A100 GPUs
- For local development, CPU inference works but is slower
- Training logs are saved in `train_*.out` files
- Model checkpoints are in `checkpoints/`
- Evaluation results are in `results/`

---

This was a project for a university machine learning course. Feel free to experiment with the code or use it as a starting point for your own projects!
