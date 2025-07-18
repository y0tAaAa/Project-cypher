#!/bin/bash

# Активация виртуального окружения (если есть)
source venv/bin/activate

# Генерация данных
echo "Generating training data..."
python src/data_gen.py

# Запуск обучения с nohup для работы в фоновом режиме
echo "Starting model training..."
nohup python src/train_model_variants.py > training.log 2>&1 &

echo "Training process started in background. You can monitor progress with:"
echo "tail -f training.log" 