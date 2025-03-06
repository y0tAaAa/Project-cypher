# Project Cypher

# Cryptanalysis with LLM (Large Language Models)

This project investigates the capabilities of neural network models (Large Language Models - LLMs), such as GPT-2 or GPT-Neo, for cryptanalysis of classical ciphers including Caesar cipher, monoalphabetic substitution cipher, Vigenère cipher, and transposition ciphers.

## Project structure

```
data/
├── train.csv
├── val.csv
├── test.csv
├── caesar_pairs.csv
├── substitution_pairs.csv
├── vigenere_pairs.csv
└── transposition_pairs.csv

jupiter_notebooks/
└── analysis.ipynb

src/
├── cipher.py
├── data_gen.py
├── decryptor.py
└── train_model.py

requirements.txt
README.md
```

## Installation

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Project Structure

- **src/cipher.py**: Functions for encryption and decryption.
- **src/data_gen.py**: Script for generating synthetic training data.
- **src/decryptor.py**: Decryption module using neural network models.
- **src/train_model.py**: Model fine-tuning script.
- **notebooks/**: Interactive Jupyter notebooks for analysis and results visualization.

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Generate data
```bash
python src/data_gen.py
```

### Train the model
```bash
python src/train_model.py
```

### Run Decryption
```bash
python src/decryptor.py
```

## Technologies
- Python
- PyTorch
- Hugging Face Transformers (GPT-2, GPT-Neo)

## Evaluation Metrics
- Decryption accuracy (character-level)
- Processing time per text

## Further Improvements
- Enhancing model performance through different architectures and hyperparameter tuning.
- Extending the approach to additional cipher types and historical datasets.


# Криптоанализ с применением нейросетей

## Описание проекта
Данный проект посвящён исследованию возможностей нейросетевых моделей (Large Language Models – LLM), таких как GPT-2 или GPT-Neo, для криптоанализа классических шифров: шифра Цезаря, моноалфавитной замены, шифра Виженера и транспозиционных шифров.

## Структура проекта
```
project/
├── data                   # Папка для хранения датасетов
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── caesar_pairs.csv
├── jupiter_notebooks
│   └── analysis.ipynb
├── src
│   ├── cipher.py
│   ├── data_gen.py
│   ├── decryptor.py
│   └── train_model.py
├── requirements.txt
└── README.md

## Установка зависимостей
Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Структура проекта
- **src/cipher.py**: Функции для шифрования и дешифрования.
- **src/data_gen.py**: Генерация синтетических данных для обучения.
- **src/decryptor.py**: Модуль дешифрования с использованием нейросетевых моделей.
- **notebooks/**: Jupyter Notebook для анализа и визуализации результатов.
- **data/**: Хранение данных, используемых для обучения и тестирования.

## Запуск проекта

### Генерация данных
```bash
python src/data_gen.py
```

### Обучение модели
```bash
python src/train_model.py
```

### Тестирование дешифровки
```bash
python src/decryptor.py
```

## Используемые технологии
- Python
- PyTorch
- Hugging Face Transformers (GPT-2, GPT-Neo)
- Datasets, Pandas, NumPy, Matplotlib

## Ноутбуки (Jupyter Notebooks)
Используются для анализа и визуализации результатов экспериментов. Все этапы (генерация данных, обучение модели, оценка точности и визуализация) документируются в интерактивном формате.

## Метрики оценки
- Точность расшифровки (посимвольная)
- Время обработки текстов

## Дальнейшее развитие
- Улучшение модели на основе разных архитектур и настроек.
- Добавление других типов шифров
- Реализация идентификатора типа шифра

## Зависимости
Указаны в `requirements.txt`, включают:
- transformers
- torch
- datasets
- pandas
- numpy
- scikit-learn
- matplotlib

