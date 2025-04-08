# src/train_model.py
import os
# Ограничиваем видимость GPU только для устройства 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Класс для датасета с криптопарами
class CipherDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):  # уменьшенное значение max_length для экономии памяти
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts[idx]
        encoding = self.tokenizer(
            item,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Функция для загрузки и форматирования данных
def load_and_format_data(file_path):
    df = pd.read_csv(file_path)
    formatted_data = []
    for _, row in df.iterrows():
        formatted_text = f"Ciphertext: {row['ciphertext']}\nPlaintext: {row['plaintext']}"
        formatted_data.append(formatted_text)
    return formatted_data

def main():
    model_name = "gpt2"  # базовая модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 не имеет отдельного pad_token, поэтому используем eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Включаем gradient checkpointing для экономии памяти
    model.gradient_checkpointing_enable()
    # Отключаем кеширование, несовместимое с checkpointing
    model.config.use_cache = False

    # Загружаем тренировочные и валидационные данные
    train_texts = load_and_format_data("data/train.csv")
    val_texts = load_and_format_data("data/val.csv")

    train_dataset = CipherDataset(train_texts, tokenizer)
    val_dataset = CipherDataset(val_texts, tokenizer)

    # Настройка тренинговых аргументов с минимальными размерами батча
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,    # batch size 1 для экономии памяти
        per_device_eval_batch_size=1,     # batch size 1 для оценки
        gradient_accumulation_steps=2,    # эффективный batch size = 2
        eval_strategy="steps",
        eval_steps=500,                   # оценка реже, чтобы снизить нагрузку
        logging_steps=100,
        save_steps=1000,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        fp16=True                       # использование fp16 для экономии видеопамяти
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Запуск тренировки
    trainer.train()

    # Создаем директорию для сохранения дообученной модели, если её нет
    os.makedirs("fine_tuned_model", exist_ok=True)
    trainer.save_model("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")

    print("✅ Model fine-tuned and saved to 'fine_tuned_model/'")

if __name__ == "__main__":
    main()
