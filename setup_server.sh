#!/bin/bash

# Создаем виртуальное окружение
python -m virtualenv ./venv

# Активируем окружение
source ./venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Проверяем CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU names:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

echo "Setup completed. Use 'source ./venv/bin/activate' to activate the environment." 