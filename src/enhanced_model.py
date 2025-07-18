import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GPTNeoXConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PeftConfig,
)
import bitsandbytes as bnb
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from accelerate import Accelerator
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    base_model_name: str = "EleutherAI/gpt-neox-20b"  # Большая базовая модель
    model_max_length: int = 1024  # Максимальная длина последовательности
    gradient_checkpointing: bool = True  # Экономия памяти
    torch_dtype: torch.dtype = torch.bfloat16  # Более эффективный формат
    load_in_8bit: bool = True  # Квантизация для экономии памяти
    device_map: str = "auto"
    
    # Конфигурация LoRA для эффективной тонкой настройки
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Ранг адаптации (больше = больше параметров)
        lora_alpha=32,  # Альфа масштабирования
        lora_dropout=0.05,  # Дропаут для регуляризации
        bias="none",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )
    
    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="results/enhanced_cipher_model",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

class EnhancedCipherModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.accelerator = Accelerator()
        self.model = None
        self.tokenizer = None
        
    def load_base_model(self):
        """Загрузка и настройка базовой модели"""
        print("Loading base model and tokenizer...")
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            model_max_length=self.config.model_max_length,
            padding_side="right",
            use_fast=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Загружаем модель с оптимизациями
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=self.config.torch_dtype,
            load_in_8bit=self.config.load_in_8bit,
            device_map=self.config.device_map,
            trust_remote_code=True,
        )
        
        # Применяем LoRA адаптацию
        self.model = get_peft_model(self.model, self.config.lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, data: pd.DataFrame) -> Dataset:
        """Подготовка данных для обучения"""
        def format_example(row):
            if row['operation'] == 'encrypt':
                prompt = f"Operation: encrypt\nCipher: {row['cipher_type']}\nLanguage: {row['language']}\nInput: {row['input_text']}\nOutput:"
                completion = f" {row['output_text']}\n"
            else:
                prompt = f"Operation: decrypt\nCipher: {row['cipher_type']}\nLanguage: {row['language']}\nInput: {row['input_text']}\nOutput:"
                completion = f" {row['output_text']}\n"
            
            return {
                "text": prompt + completion,
                "prompt": prompt,
                "completion": completion
            }
        
        formatted_data = data.apply(format_example, axis=1)
        dataset_dict = {
            "text": formatted_data.apply(lambda x: x["text"]).tolist(),
            "prompt": formatted_data.apply(lambda x: x["prompt"]).tolist(),
            "completion": formatted_data.apply(lambda x: x["completion"]).tolist(),
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def train(self, train_data: pd.DataFrame, eval_data: pd.DataFrame):
        """Обучение модели"""
        print("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_data)
        eval_dataset = self.prepare_dataset(eval_data)
        
        # Создаем data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Инициализируем trainer
        trainer = Trainer(
            model=self.model,
            args=self.config.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Сохраняем модель
        print("Saving model...")
        trainer.save_model("results/enhanced_cipher_model_final")
        
    def process_text(
        self,
        text: str,
        operation: str,
        cipher_type: str,
        language: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Шифрование или дешифрование текста"""
        prompt = f"Operation: {operation}\nCipher: {cipher_type}\nLanguage: {language}\nInput: {text}\nOutput:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text.split("Output:")[-1].strip()
        
        return result

def main():
    """Пример использования"""
    # Инициализация конфигурации и модели
    config = ModelConfig()
    model = EnhancedCipherModel(config)
    
    # Загрузка базовой модели
    model.load_base_model()
    
    # Загрузка данных
    train_data = pd.read_csv("data/train_enhanced_vigenere.csv")
    eval_data = pd.read_csv("data/val_enhanced_vigenere.csv")
    
    # Обучение
    model.train(train_data, eval_data)
    
    # Пример использования
    text = "Hello, World!"
    result = model.process_text(
        text=text,
        operation="encrypt",
        cipher_type="vigenere",
        language="en"
    )
    print(f"Input: {text}")
    print(f"Encrypted: {result}")

if __name__ == "__main__":
    main() 