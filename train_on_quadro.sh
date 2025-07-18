#!/bin/bash

# Активируем виртуальное окружение
source ./venv/bin/activate

# Устанавливаем переменные окружения для CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO

# Запускаем мониторинг GPU в фоновом режиме
nvidia-smi dmon -i 0,1,2,3 -s u -o TD >> gpu_stats.log &
MONITOR_PID=$!

# Запускаем обучение
echo "Starting distributed training..."
python src/train_distributed.py 2>&1 | tee training.log

# Останавливаем мониторинг
kill $MONITOR_PID

echo "Training completed. Check training.log and gpu_stats.log for details." 