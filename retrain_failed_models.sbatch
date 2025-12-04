#!/bin/bash
#SBATCH --job-name=retrain_failed
#SBATCH --partition=nvidia
#SBATCH --constraint=80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=71:59:59
#SBATCH --mem=32GB
#SBATCH --output=/home/mom8702/Explorative-Facial-Expressions-Classifier/retrain_%j.out
#SBATCH --error=/home/mom8702/Explorative-Facial-Expressions-Classifier/retrain_%j.err

# Set environment variables to prevent threading issues
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Print GPU information
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
nvidia-smi
echo "=========================================="
echo ""

# Change to project directory
cd /home/mom8702/Explorative-Facial-Expressions-Classifier

# Activate conda environment
conda activate acl

# Verify PyTorch can see GPU
echo "Checking PyTorch GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Retrain the 3 failed models
echo "=========================================="
echo "Retraining Failed Models"
echo "=========================================="
echo "Models to retrain:"
echo "  1. hybrid_transformer (25.11% -> expected 65-72%)"
echo "  2. vit_tiny (25.22% -> expected 60-68%)"
echo "  3. multiscale (37.28% -> expected 50-55%)"
echo ""

# Model 1: Hybrid Transformer
echo ""
echo "=========================================="
echo "Training hybrid_transformer..."
echo "=========================================="
python -u scripts/train.py --model hybrid_transformer
echo "✅ hybrid_transformer training complete!"
echo ""

# Clear GPU cache between models
python -c "import torch; torch.cuda.empty_cache()"

# Model 2: ViT-Tiny
echo ""
echo "=========================================="
echo "Training vit_tiny..."
echo "=========================================="
python -u scripts/train.py --model vit_tiny
echo "✅ vit_tiny training complete!"
echo ""

# Clear GPU cache between models
python -c "import torch; torch.cuda.empty_cache()"

# Model 3: Multi-Scale
echo ""
echo "=========================================="
echo "Training multiscale..."
echo "=========================================="
python -u scripts/train.py --model multiscale
echo "✅ multiscale training complete!"
echo ""

echo ""
echo "=========================================="
echo "All 3 models retrained successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check results in checkpoints/ directory"
echo "  2. Run: python scripts/evaluate.py"
echo "  3. Compare new results with previous training"

