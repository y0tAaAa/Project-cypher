import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
from tqdm import tqdm
import torch
from typing import List, Dict, Tuple
import evaluate
import json
import os

def load_test_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load test data from CSV file."""
    df = pd.read_csv(file_path)
    return df['encrypted_text'].tolist(), df['original_text'].tolist()

def generate_decryption(model, tokenizer, encrypted_text: str, max_length: int = 100) -> str:
    """Generate decryption using the model."""
    inputs = tokenizer(encrypted_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate various metrics including BLEU."""
    metrics = {}
    
    # BLEU score
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    metrics['bleu'] = bleu_score['bleu']
    
    # ROUGE scores
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    metrics.update({k: v for k, v in rouge_scores.items()})
    
    # METEOR score
    meteor = evaluate.load('meteor')
    meteor_score = meteor.compute(predictions=predictions, references=references)
    metrics['meteor'] = meteor_score['meteor']
    
    # Character Error Rate
    cer = evaluate.load('cer')
    cer_score = cer.compute(predictions=predictions, references=references)
    metrics['cer'] = cer_score['cer']
    
    # Word Error Rate
    wer = evaluate.load('wer')
    wer_score = wer.compute(predictions=predictions, references=references)
    metrics['wer'] = wer_score['wer']
    
    return metrics

def evaluate_model(model_path: str, test_files: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Evaluate model on multiple languages."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    results = {}
    
    for lang, test_file in test_files.items():
        print(f"\nEvaluating {lang}...")
        encrypted_texts, original_texts = load_test_data(test_file)
        
        predictions = []
        for text in tqdm(encrypted_texts):
            pred = generate_decryption(model, tokenizer, text)
            predictions.append(pred)
            
        metrics = calculate_metrics(predictions, original_texts)
        results[lang] = metrics
        
        # Save predictions for manual inspection
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/{lang}_predictions.json", "w", encoding="utf-8") as f:
            json.dump({
                "predictions": predictions,
                "references": original_texts,
                "metrics": metrics
            }, f, ensure_ascii=False, indent=2)
            
    return results

def main():
    # Define test files for each language
    test_files = {
        "english": "data/test_en.csv",
        "slovak": "data/test_sk.csv",
        "ukrainian": "data/test_uk.csv"
    }
    
    # List of models to evaluate
    models = [
        "y0ta/fine_tuned_model",  # Original model
        "y0ta/multilingual_model", # Multilingual variant
        "y0ta/gpt2_large_model"   # Larger model variant
    ]
    
    all_results = {}
    
    for model_path in models:
        print(f"\nEvaluating model: {model_path}")
        results = evaluate_model(model_path, test_files)
        all_results[model_path] = results
    
    # Create comparison table
    comparison_table = []
    for model_name, model_results in all_results.items():
        for lang, metrics in model_results.items():
            row = {
                "Model": model_name,
                "Language": lang,
                **metrics
            }
            comparison_table.append(row)
    
    # Save comparison table
    df = pd.DataFrame(comparison_table)
    df.to_csv("evaluation_results/comparison_table.csv", index=False)
    print("\nResults saved to evaluation_results/comparison_table.csv")

if __name__ == "__main__":
    main() 