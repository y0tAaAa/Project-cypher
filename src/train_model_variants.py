import os
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict
import json

class CipherDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
    
    def __len__(self):
        return len(self.encodings.input_ids)

def load_and_format_data(file_path: str) -> List[str]:
    """Load and format data from CSV file."""
    df = pd.read_csv(file_path)
    # Format as input-output pairs
    texts = []
    for _, row in df.iterrows():
        text = f"Encrypted: {row['encrypted_text']}\nDecrypted: {row['original_text']}"
        texts.append(text)
    return texts

def train_model_variant(
    model_name: str,
    train_files: Dict[str, str],
    val_files: Dict[str, str],
    output_dir: str,
    training_args: Dict
) -> None:
    """Train a model variant with specified parameters."""
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # Load and combine training data from all languages
    train_texts = []
    for lang, file_path in train_files.items():
        texts = load_and_format_data(file_path)
        train_texts.extend(texts)
    
    # Load and combine validation data
    val_texts = []
    for lang, file_path in val_files.items():
        texts = load_and_format_data(file_path)
        val_texts.extend(texts)
    
    # Create datasets
    train_dataset = CipherDataset(train_texts, tokenizer)
    val_dataset = CipherDataset(val_texts, tokenizer)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        **training_args
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model trained and saved to '{output_dir}'")

def main():
    # Base model configurations
    model_variants = {
        "multilingual": {
            "model_name": "gpt2",
            "training_args": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "eval_steps": 500,
                "save_steps": 1000,
                "logging_steps": 100,
                "fp16": True
            }
        },
        "gpt2_large": {
            "model_name": "gpt2-large",
            "training_args": {
                "num_train_epochs": 2,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": 3e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "eval_steps": 500,
                "save_steps": 1000,
                "logging_steps": 100,
                "fp16": True
            }
        }
    }
    
    # Data files
    languages = ["english", "slovak", "ukrainian"]
    train_files = {lang: f"data/train_{lang[:2]}.csv" for lang in languages}
    val_files = {lang: f"data/val_{lang[:2]}.csv" for lang in languages}
    
    # Train each model variant
    for variant_name, config in model_variants.items():
        print(f"\nTraining {variant_name} model...")
        output_dir = f"models/{variant_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        train_model_variant(
            model_name=config["model_name"],
            train_files=train_files,
            val_files=val_files,
            output_dir=output_dir,
            training_args=config["training_args"]
        )

if __name__ == "__main__":
    main() 