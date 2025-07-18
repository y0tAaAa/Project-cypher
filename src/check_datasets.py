import pandas as pd
import os

def analyze_dataset(file_path):
    """Analyze dataset size and distribution"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    total = len(df)
    
    print(f"\nDataset: {os.path.basename(file_path)}")
    print(f"Total examples: {total:,}")
    
    # Распределение по языкам
    print("\nDistribution by language:")
    lang_dist = df['language'].value_counts()
    for lang, count in lang_dist.items():
        print(f"{lang}: {count:,} ({count/total*100:.1f}%)")
    
    # Распределение по операциям
    print("\nDistribution by operation:")
    op_dist = df['operation'].value_counts()
    for op, count in op_dist.items():
        print(f"{op}: {count:,} ({count/total*100:.1f}%)")

# Анализируем все датасеты
datasets = [
    'data/train_enhanced_vigenere.csv',
    'data/val_enhanced_vigenere.csv',
    'data/test_enhanced_vigenere.csv',
    'data/train_enhanced_substitution.csv',
    'data/val_enhanced_substitution.csv',
    'data/test_enhanced_substitution.csv'
]

for dataset in datasets:
    analyze_dataset(dataset)

# Показываем общий размер
total_examples = sum(len(pd.read_csv(f)) for f in datasets if os.path.exists(f))
print(f"\nTotal examples across all datasets: {total_examples:,}") 