#!/bin/bash
# Helper script to run setup_data.py with proper environment variables for HPC

# Set OpenBLAS threading to prevent resource limit issues on login nodes
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run the setup script
python scripts/setup_data.py "$@"

