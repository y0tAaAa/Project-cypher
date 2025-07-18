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

# Training parameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=3
LEARNING_RATE=2e-5

echo "Starting training with the following configuration:"
echo "- Project directory: $PROJECT_DIR"
echo "- Batch size: $BATCH_SIZE"
echo "- Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "- Number of epochs: $NUM_EPOCHS"
echo "- Learning rate: $LEARNING_RATE"
echo "- Using 4 GPUs for distributed training"
echo "- Training logs will be saved to: training.log"
echo "- GPU monitoring logs will be saved to: gpu_monitoring.log"
echo

# Check if training script exists
if [ ! -f "src/train_model_variants.py" ]; then
    echo "ERROR: Training script not found at src/train_model_variants.py"
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

# Start distributed training
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    src/train_model_variants.py \
    --model_name "EleutherAI/gpt-neo-2.7B" \
    --train_file "data/train_en.csv,data/train_uk.csv,data/train_sl.csv" \
    --validation_file "data/val_en.csv,data/val_uk.csv,data/val_sl.csv" \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --fp16 \
    --use_8bit_adam \
    --use_lora \
    --save_strategy "steps" \
    --save_steps 5000 \
    --logging_steps 100 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --warmup_steps 100 \
    --output_dir "./results" \
    --overwrite_output_dir \
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