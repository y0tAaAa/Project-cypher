import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tqdm import tqdm

class CipherDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
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

def load_and_format_data(cipher_files):
    formatted_data = []
    
    for cipher_type, file_path in cipher_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue
            
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            formatted_text = f"Operation: {row['operation']}\nCipher: {cipher_type}\nInput: {row['input_text']}\nOutput: {row['output_text']}"
            formatted_data.append(formatted_text)
    
    return formatted_data

def main():
    # Use a more powerful model
    model_name = "gpt2-medium"
    
    print(f"Loading {model_name} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Define files for each cipher type
    train_files = {
        'vigenere': 'data/train_enhanced_vigenere.csv',
        'substitution': 'data/train_enhanced_substitution.csv'
    }
    
    val_files = {
        'vigenere': 'data/val_enhanced_vigenere.csv',
        'substitution': 'data/val_enhanced_substitution.csv'
    }

    print("Loading and formatting training data...")
    train_texts = load_and_format_data(train_files)
    
    print("Loading and formatting validation data...")
    val_texts = load_and_format_data(val_files)

    print("Creating datasets...")
    train_dataset = CipherDataset(train_texts, tokenizer)
    val_dataset = CipherDataset(val_texts, tokenizer)

    # Configure training parameters
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        warmup_steps=1000,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    output_dir = "fine_tuned_model_multi"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Model fine-tuned and saved to '{output_dir}/'")

if __name__ == "__main__":
    main() 