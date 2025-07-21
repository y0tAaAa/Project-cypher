#!/bin/bash

# Check if PROJECT_DIR is set
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: PROJECT_DIR environment variable is not set"
    exit 1
fi

# Ensure we're in the project directory
cd $PROJECT_DIR || exit 1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher || exit 1

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="cypher-training"

echo "Starting training with the following configuration:"
echo "- Project directory: $PROJECT_DIR"
echo "- Using 4 GPUs for distributed training"
echo "- Training logs will be saved to: training.log"
echo "- GPU monitoring logs will be saved to: gpu_monitoring.log"
echo

# Check if training script exists
if [ ! -f "./src/train_model_variants.py" ]; then
    echo "ERROR: Training script not found at ./src/train_model_variants.py"
    exit 1
fi

# Monitor GPU usage
echo "Starting GPU monitoring..."
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu --format=csv -l 60 > gpu_monitoring.log &
MONITOR_PID=$!

# Trap to kill GPU monitoring on script exit
trap "kill $MONITOR_PID" EXIT

echo "Initial GPU state:"
nvidia-smi

echo -e "\nStarting distributed training...\n"

# Start distributed training with relative path
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    ./src/train_model_variants.py \
    2>&1 | tee training.log

training_status=$?

if [ $training_status -eq 0 ]; then
    echo -e "\nTraining completed successfully!"
else
    echo -e "\nTraining failed with exit code $training_status"
    echo "Check training.log for error details"
fi

echo -e "\nFinal GPU state:"
nvidia-smi

echo -e "\nTraining logs are available in: training.log"
echo "GPU monitoring logs are available in: gpu_monitoring.log" 