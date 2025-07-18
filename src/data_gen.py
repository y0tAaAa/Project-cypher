# src/data_gen.py
import random
import string
import pandas as pd
from typing import List, Tuple, Dict
import os
from tqdm import tqdm

# Расширенный набор символов для разных языков
ALPHABETS = {
    'en': list(string.ascii_uppercase),
    'uk': list("АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"),
    'sl': list("AÁÄBCČDĎDZDŽEÉFGHCHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ")
}

def load_existing_datasets(base_path: str = 'data') -> Dict[str, pd.DataFrame]:
    """Load existing datasets for all languages"""
    datasets = {}
    languages = ['en', 'uk', 'sl']
    types = ['train', 'val', 'test']
    
    for lang in languages:
        for dtype in types:
            file_path = os.path.join(base_path, f'{dtype}_{lang}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                datasets[f'{dtype}_{lang}'] = df
                print(f"✓ Loaded {file_path}")
    
    return datasets

def vigenere_encrypt(text: str, key: str, alphabet: List[str]) -> str:
    """Encrypt text using Vigenere cipher with custom alphabet"""
    encrypted = ''
    key = key.upper()
    text = text.upper()
    alphabet_size = len(alphabet)
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    for i, char in enumerate(text):
        if char in char_to_idx:
            key_char = key[i % len(key)]
            key_idx = char_to_idx.get(key_char, 0)
            char_idx = char_to_idx[char]
            new_idx = (char_idx + key_idx) % alphabet_size
            encrypted += alphabet[new_idx]
        else:
            encrypted += char
    return encrypted

def vigenere_decrypt(text: str, key: str, alphabet: List[str]) -> str:
    """Decrypt text using Vigenere cipher with custom alphabet"""
    decrypted = ''
    key = key.upper()
    text = text.upper()
    alphabet_size = len(alphabet)
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    for i, char in enumerate(text):
        if char in char_to_idx:
            key_char = key[i % len(key)]
            key_idx = char_to_idx.get(key_char, 0)
            char_idx = char_to_idx[char]
            new_idx = (char_idx - key_idx) % alphabet_size
            decrypted += alphabet[new_idx]
        else:
            decrypted += char
    return decrypted

def substitution_encrypt(text: str, alphabet: List[str]) -> Tuple[str, Dict[str, str]]:
    """Encrypt text using monoalphabetic substitution cipher with custom alphabet"""
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    mapping = dict(zip(alphabet, shuffled))
    
    encrypted = ''
    text = text.upper()
    for char in text:
        if char in mapping:
            encrypted += mapping[char]
        else:
            encrypted += char
    return encrypted, mapping

def substitution_decrypt(text: str, mapping: Dict[str, str]) -> str:
    """Decrypt text using monoalphabetic substitution cipher"""
    reverse_mapping = {v: k for k, v in mapping.items()}
    decrypted = ''
    for char in text:
        if char in reverse_mapping:
            decrypted += reverse_mapping[char]
        else:
            decrypted += char
    return decrypted

def generate_enhanced_dataset(datasets: Dict[str, pd.DataFrame], cipher_type: str) -> List[Tuple[str, str, str, str, str]]:
    """Generate enhanced dataset with both encryption and decryption examples"""
    enhanced_data = []
    keys = {
        'en': ['PYTHON', 'CRYPTO', 'SECRET', 'CIPHER', 'SECURE'],
        'uk': ['ШИФР', 'КЛЮЧ', 'КРИПТО', 'БЕЗПЕКА', 'ЗАХИСТ'],
        'sl': ['SIFRA', 'KLUC', 'KRYPTO', 'BEZPECNOST', 'OCHRANA']
    }
    
    for dataset_name, df in datasets.items():
        lang = dataset_name.split('_')[1]  # en, uk, or sl
        alphabet = ALPHABETS[lang]
        lang_keys = keys[lang]
        print(f"\nProcessing {dataset_name}...")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            original_text = row['original_text']
            existing_encrypted = row['encrypted_text']
            
            # Добавляем пример из существующего датасета (дешифровка)
            enhanced_data.append((
                existing_encrypted,
                original_text,
                'decrypt',
                cipher_type,
                lang
            ))
            
            # Создаем новый пример шифрования
            if cipher_type == 'vigenere':
                key = random.choice(lang_keys)
                new_encrypted = vigenere_encrypt(original_text, key, alphabet)
                enhanced_data.append((
                    original_text,
                    new_encrypted,
                    'encrypt',
                    cipher_type,
                    lang
                ))
                
            elif cipher_type == 'substitution':
                new_encrypted, _ = substitution_encrypt(original_text, alphabet)
                enhanced_data.append((
                    original_text,
                    new_encrypted,
                    'encrypt',
                    cipher_type,
                    lang
                ))
    
    return enhanced_data

def save_enhanced_dataset(dataset: List[Tuple[str, str, str, str, str]], base_name: str):
    """Save enhanced dataset to CSV files"""
    df = pd.DataFrame(dataset, columns=['input_text', 'output_text', 'operation', 'cipher_type', 'language'])
    
    # Разделяем на train/val/test с сохранением пропорций языков
    train_ratio, val_ratio = 0.7, 0.15  # test_ratio будет 0.15
    
    # Сначала разделяем по языкам
    languages = df['language'].unique()
    train_data, val_data, test_data = [], [], []
    
    for lang in languages:
        lang_data = df[df['language'] == lang].values.tolist()
        n_samples = len(lang_data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Перемешиваем данные
        random.shuffle(lang_data)
        
        train_data.extend(lang_data[:n_train])
        val_data.extend(lang_data[n_train:n_train + n_val])
        test_data.extend(lang_data[n_train + n_val:])
    
    # Создаем и сохраняем DataFrame'ы
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        df = pd.DataFrame(data, columns=['input_text', 'output_text', 'operation', 'cipher_type', 'language'])
        filename = f'data/{name}_{base_name}.csv'
        df.to_csv(filename, index=False)
        print(f"✓ Saved {len(df)} examples to {filename}")

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load existing datasets
    print("Loading existing datasets...")
    datasets = load_existing_datasets()
    
    if not datasets:
        print("No existing datasets found!")
        return
    
    # Generate enhanced datasets for each cipher type
    for cipher_type in ['vigenere', 'substitution']:
        print(f"\nGenerating enhanced dataset for {cipher_type} cipher...")
        enhanced_data = generate_enhanced_dataset(datasets, cipher_type)
        save_enhanced_dataset(enhanced_data, f'enhanced_{cipher_type}')

if __name__ == "__main__":
    main()
