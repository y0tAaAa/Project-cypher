# src/cypher.py
import random
import string
import math
from typing import Tuple, Dict

# Caesar cipher
def caesar_encrypt(text: str, shift: int = 3) -> str:
    encrypted = ''
    for char in text.upper():
        if char.isalpha():
            encrypted += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            encrypted += char
    return encrypted

def caesar_decrypt(text: str, shift: int = 3) -> str:
    return caesar_encrypt(text, -shift)

# Monoalphabetic substitution cipher
def substitution_encrypt(text: str) -> Tuple[str, Dict[str, str]]:
    alphabet = list(string.ascii_uppercase)
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    mapping = dict(zip(alphabet, shuffled))
    encrypted = ''.join(mapping.get(c, c) for c in text.upper())
    return encrypted, mapping

def substitution_decrypt(text: str, mapping: dict) -> str:
    reverse_mapping = {v: k for k, v in mapping.items()}
    return ''.join(reverse_mapping.get(char, char) for char in text)

# Vigenère cipher
def vigenere_encrypt(text: str, key: str) -> str:
    encrypted = ''
    key = key.upper()
    text = text.upper().replace(" ", "")
    for i, char in enumerate(text):
        if char.isalpha():
            shift = ord(key[i % len(key)]) - ord('A')
            encrypted += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            encrypted += char
    return encrypted

def vigenere_decrypt(text: str, key: str) -> str:
    decrypted = ''
    key = key.upper()
    key_len = len(key)
    text = text.upper().replace(" ", "")
    for i, char in enumerate(text):
        if char.isalpha():
            shift = ord(key[i % len(key)]) - ord('A')
            decrypted += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
        else:
            decrypted += char
    return decrypted

# Columnar Transposition cipher
def columnar_transposition_encrypt(text: str, key: int) -> str:
    text = text.replace(" ", "")
    n_rows = math.ceil(len(text) / key)
    padded_text = text.ljust(n_rows * key, 'X')
    matrix = [padded_text[i*key:(i+1)*key] for i in range(n_rows)]
    encrypted = ''.join(''.join(row[i] for row in matrix if i < len(row)) for i in range(key))
    return encrypted
