import pandas as pd
import random
import string
from typing import List, Tuple
import os

def caesar_encrypt(text: str, shift: int) -> str:
    """Encrypt text using Caesar cipher."""
    result = ""
    # Handle both lowercase and uppercase letters
    for char in text:
        if char.isalpha():
            # Determine the case and base ASCII value
            ascii_base = ord('A') if char.isupper() else ord('a')
            # Apply shift and wrap around
            shifted = (ord(char) - ascii_base + shift) % 26
            result += chr(ascii_base + shifted)
        else:
            result += char
    return result

def load_language_texts(language: str) -> List[str]:
    """Load sample texts for a given language."""
    if language == "english":
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "To be or not to be that is the question",
            "All that glitters is not gold",
            "A journey of a thousand miles begins with a single step",
            "Actions speak louder than words",
            "Beauty is in the eye of the beholder",
            "Every cloud has a silver lining",
            "Fortune favors the bold",
            "Knowledge is power",
            "Time heals all wounds"
        ]
    elif language == "slovak":
        texts = [
            "Príliš žlťučký kôň úpäl ďábelské ódy",
            "Kde bolo tam bolo",
            "V našej záhrade rastú krásne kvety",
            "Všetko najlepšie k narodeninám",
            "Dobrý deň prajem",
            "Slovensko je krásna krajina",
            "Život je krásny",
            "Láska hory prenáša",
            "Ráno múdrejšie večera",
            "Bez práce nie sú koláče"
        ]
    elif language == "ukrainian":
        texts = [
            "Жебракують філософи при ґанку церкви в Гадячі",
            "Щастя в боротьбі",
            "Чуєш їхній голос",
            "Доброго дня",
            "Слава Україні",
            "Життя прекрасне",
            "Все буде добре",
            "Мир у всьому світі",
            "Любов врятує світ",
            "Знання це сила"
        ]
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    return texts

def generate_data(language: str, num_samples: int) -> Tuple[List[str], List[str]]:
    """Generate data pairs (original, encrypted) for a language."""
    texts = load_language_texts(language)
    original_texts = []
    encrypted_texts = []
    
    # Generate samples by combining and modifying base texts
    while len(original_texts) < num_samples:
        # Take 1-3 random texts and combine them
        num_texts = random.randint(1, 3)
        selected_texts = random.sample(texts, num_texts)
        combined_text = " ".join(selected_texts)
        
        # Random shift between 1 and 25
        shift = random.randint(1, 25)
        encrypted = caesar_encrypt(combined_text, shift)
        
        original_texts.append(combined_text)
        encrypted_texts.append(encrypted)
    
    return original_texts, encrypted_texts

def main():
    # Parameters
    languages = ["english", "slovak", "ukrainian"]
    train_samples = 5000  # Number of training samples per language
    val_samples = 1000    # Number of validation samples per language
    test_samples = 1000   # Number of test samples per language
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate data for each language
    for lang in languages:
        print(f"Generating data for {lang}...")
        
        # Generate training data
        print(f"  Generating {train_samples} training samples...")
        train_texts, train_encrypted = generate_data(lang, train_samples)
        train_df = pd.DataFrame({
            'original_text': train_texts,
            'encrypted_text': train_encrypted
        })
        train_df.to_csv(f"data/train_{lang[:2]}.csv", index=False)
        
        # Generate validation data
        print(f"  Generating {val_samples} validation samples...")
        val_texts, val_encrypted = generate_data(lang, val_samples)
        val_df = pd.DataFrame({
            'original_text': val_texts,
            'encrypted_text': val_encrypted
        })
        val_df.to_csv(f"data/val_{lang[:2]}.csv", index=False)
        
        # Generate test data
        print(f"  Generating {test_samples} test samples...")
        test_texts, test_encrypted = generate_data(lang, test_samples)
        test_df = pd.DataFrame({
            'original_text': test_texts,
            'encrypted_text': test_encrypted
        })
        test_df.to_csv(f"data/test_{lang[:2]}.csv", index=False)
        
        print(f"✅ {lang} data generation complete")

if __name__ == "__main__":
    main() 