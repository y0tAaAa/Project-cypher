#project Cypher
# cypher.py
import random
import string
import math

# Шифр Цезаря
def caesar_encrypt(text: str, shift: int = 3) -> str:
    encrypted = ''
    for char in text.upper():
        if char.isalpha():
            encrypted += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            encrypted += char
    return encrypted

# Моноалфавитная замена
def substitution_encrypt(text: str) -> (str, dict):
    alphabet = list(string.ascii_uppercase)
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    mapping = dict(zip(alphabet, shuffled))
    encrypted = ''.join(mapping.get(c, c) for c in text.upper())
    return encrypted, mapping

# Шифр Виженера
def vigenere_encrypt(text: str, key: str) -> str:
    encrypted = ''
    key = key.upper()
    key_len = len(key)
    text = text.upper().replace(" ", "")
    for i, char in enumerate(text):
        if char.isalpha():
            shift = ord(key[i % len(key)]) - ord('A')
            encrypted += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            encrypted += char
    return encrypted

# Транспозиционный шифр (колоночная транспозиция)
def columnar_transposition_encrypt(text: str, key: int) -> str:
    text = text.replace(" ", "")
    n_rows = math.ceil(len(text) / key)
    padded_text = text.ljust(n * key, 'X')
    matrix = [text[i*key:(i+1)*key] for i in range(n)]
    encrypted = ''.join(''.join(row[i] for row in matrix if i < len(row)) for i in range(key))
    return encrypted
