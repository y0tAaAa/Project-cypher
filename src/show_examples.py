import pandas as pd
import random

def show_examples(file_path, n_examples=3):
    """Show random examples from dataset"""
    df = pd.read_csv(file_path)
    print(f"\nExamples from {file_path}:")
    print(f"Total examples: {len(df)}")
    
    # Get examples for each language and operation
    for lang in df['language'].unique():
        print(f"\n=== Language: {lang} ===")
        for operation in ['encrypt', 'decrypt']:
            examples = df[(df['language'] == lang) & (df['operation'] == operation)]
            if len(examples) > 0:
                sample = examples.sample(n=min(n_examples, len(examples)))
                print(f"\n--- {operation.capitalize()} examples ---")
                for _, row in sample.iterrows():
                    print(f"\nInput:  {row['input_text']}")
                    print(f"Output: {row['output_text']}")

# Показываем примеры для обоих шифров
show_examples('data/train_enhanced_vigenere.csv')
show_examples('data/train_enhanced_substitution.csv') 