#sosi penis
# src/train_model.py
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Dataset class for cipher pairs
class CipherDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
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

# Load and format dataset
def load_and_format_data(file_path):
    df = pd.read_csv(file_path)
    formatted_data = []
    for _, row in df.iterrows():
        formatted_text = f"Ciphertext: {row['ciphertext']}\nPlaintext: {row['plaintext']}"
        formatted_data.append(formatted_text)
    return formatted_data

def main():
    model_name = "gpt2"  # base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # добавляем эту строку
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load training and validation data
    train_texts = load_and_format_data("data/train.csv")
    val_texts = load_and_format_data("data/val.csv")

    train_dataset = CipherDataset(train_texts, tokenizer)
    val_dataset = CipherDataset(val_texts, tokenizer)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        save_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        loss_type = 'ForCausalLMLoss'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    trainer.train()

    # Save fine-tuned model
    trainer.save_model("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")

    print("✅ Model fine-tuned and saved to 'fine_tuned_model/'")

if __name__ == "__main__":
    main()
