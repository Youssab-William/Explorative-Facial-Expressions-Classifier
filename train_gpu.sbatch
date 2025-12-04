#!/bin/bash
#SBATCH --job-name=emotion_train
#SBATCH --partition=nvidia
#SBATCH --constraint=80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=71:59:59
#SBATCH --mem=32GB
#SBATCH --output=/home/mom8702/Explorative-Facial-Expressions-Classifier/train_%j.out
#SBATCH --error=/home/mom8702/Explorative-Facial-Expressions-Classifier/train_%j.err
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

# Activate conda environment (if using conda)
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate acl

# Or if using virtual environment
# source venv/bin/activate

# Verify PyTorch can see GPU
echo "Checking PyTorch GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run training
echo "Starting training on GPU..."
python -u scripts/train.py --model all

echo ""
echo "Training completed!"

