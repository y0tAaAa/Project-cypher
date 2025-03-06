# src/data_gen.py
import pandas as pd
import string
from datasets import load_dataset
from cypher import caesar_encrypt, substitution_encrypt, vigenere_encrypt, columnar_transposition_encrypt

# Load dataset (WikiText-2 as example)
def load_texts():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return dataset["text"]

# Clean text function
def clean_text(text: str) -> str:
    allowed_chars = set(string.ascii_uppercase + ' ')
    cleaned = ''.join([c.upper() if c.upper() in allowed_chars else ' ' for c in text])
    return ' '.join(cleaned.split())

# Generate Caesar cipher dataset
def generate_caesar_data(texts, shift=3, length=200):
    data = []
    for text in texts:
        cleaned = clean_text(text)
        if len(cleaned) < 10:
            continue
        for i in range(0, len(cleaned), length):
            fragment = cleaned[i:i+length]
            encrypted = caesar_encrypt(fragment, shift)
            data.append({'plaintext': fragment, 'ciphertext': encrypted})
    return data

# Generate substitution cipher dataset
def generate_substitution_data(texts, length=200):
    data = []
    for text in texts:
        cleaned = clean_text(text)
        if len(cleaned) < 10:
            continue
        for i in range(0, len(cleaned), length):
            fragment = cleaned[i:i+length]
            encrypted, mapping = substitution_encrypt(fragment)
            data.append({'plaintext': fragment, 'ciphertext': encrypted, 'mapping': mapping})
    return data

# Generate Vigenère cipher dataset
def generate_vigenere_data(texts, key='SECRET', length=200):
    data = []
    for text in texts:
        cleaned = clean_text(text)
        cleaned_no_spaces = cleaned.replace(" ", "")
        if len(cleaned_no_spaces) < 10:
            continue
        for i in range(0, len(cleaned_no_spaces), length):
            fragment = cleaned_no_spaces[i:i+length]
            encrypted = vigenere_encrypt(fragment, key)
            data.append({'plaintext': fragment, 'ciphertext': encrypted, 'key': key})
    return data

# Generate Columnar Transposition cipher dataset
def generate_transposition_data(texts, key=5, length=200):
    data = []
    for text in texts:
        cleaned = clean_text(text)
        cleaned_no_spaces = cleaned.replace(" ", "")
        if len(cleaned_no_spaces) < 10:
            continue
        for i in range(0, len(cleaned_no_spaces), length):
            fragment = cleaned_no_spaces[i:i+length]
            encrypted = columnar_transposition_encrypt(fragment, key)
            data.append({'plaintext': fragment, 'ciphertext': encrypted, 'key': key})
    return data

# Main function to generate datasets
def main():
    texts = load_texts()

    # Generate and save Caesar data
    caesar_data = generate_caesar_data(texts)
    pd.DataFrame(caesar_data).to_csv("data/caesar_pairs.csv", index=False)
    print("✅ Caesar cipher data saved to data/caesar_pairs.csv")

    # Generate and save substitution data
    substitution_data = generate_substitution_data(texts)
    pd.DataFrame(substitution_data).to_csv("data/substitution_pairs.csv", index=False)
    print("✅ Substitution cipher data saved to data/substitution_pairs.csv")

    # Generate and save Vigenère data
    vigenere_data = generate_vigenere_data(texts)
    pd.DataFrame(vigenere_data).to_csv("data/vigenere_pairs.csv", index=False)
    print("✅ Vigenère cipher data saved to data/vigenere_pairs.csv")

    # Generate and save Columnar Transposition data
    transposition_data = generate_transposition_data(texts)
    pd.DataFrame(transposition_data).to_csv("data/transposition_pairs.csv", index=False)
    print("✅ Columnar Transposition cipher data saved to data/transposition_pairs.csv")

if __name__ == "__main__":
    main()
