import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional, Tuple
import os

# Импорты для LLM
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CipherProcessor:
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация процессора шифрования/дешифрования.
        :param model_path: Путь к обученной модели (если используется LLM).
        """
        self.model = None
        self.tokenizer = None

        # Инициализация LLM, если указан model_path и библиотека transformers доступна
        if model_path and TRANSFORMERS_AVAILABLE:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                self.model = None
                self.tokenizer = None

    def process(self, text: str, operation: str, cipher_type: str, max_length: int = 100) -> Tuple[str, float]:
        """
        Обработка текста (шифрование или дешифрование).
        :param text: Входной текст
        :param operation: Операция ('encrypt' или 'decrypt')
        :param cipher_type: Тип шифра ('vigenere' или 'substitution')
        :param max_length: Максимальная длина выходного текста
        :return: Кортеж (обработанный_текст, уверенность)
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized")

        # Формируем промпт для модели
        prompt = f"Operation: {operation}\nCipher: {cipher_type}\nInput: {text}\nOutput:"
        
        # Генерируем ответ
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1
        )
        
        # Декодируем ответ
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        processed_text = decoded_output.split("Output:")[-1].strip()
        
        # Вычисляем метрику уверенности (простая реализация)
        confidence = self._calculate_confidence(text, processed_text)
        
        return processed_text, confidence

    def encrypt(self, plaintext: str, cipher_type: str, max_length: int = 100) -> Tuple[str, float]:
        """
        Шифрование текста.
        :param plaintext: Исходный текст
        :param cipher_type: Тип шифра
        :param max_length: Максимальная длина выходного текста
        :return: Кортеж (зашифрованный_текст, уверенность)
        """
        return self.process(plaintext, 'encrypt', cipher_type, max_length)

    def decrypt(self, ciphertext: str, cipher_type: str, max_length: int = 100) -> Tuple[str, float]:
        """
        Дешифрование текста.
        :param ciphertext: Зашифрованный текст
        :param cipher_type: Тип шифра
        :param max_length: Максимальная длина выходного текста
        :return: Кортеж (расшифрованный_текст, уверенность)
        """
        return self.process(ciphertext, 'decrypt', cipher_type, max_length)

    def _calculate_confidence(self, input_text: str, output_text: str) -> float:
        """
        Вычисляет уверенность модели в результате (простая метрика).
        :param input_text: Входной текст
        :param output_text: Выходной текст
        :return: Значение уверенности (0-1)
        """
        # Простая эвристика: проверяем, что выходной текст имеет разумную длину
        # и содержит только допустимые символы
        if not output_text or len(output_text) < 3:
            return 0.0
            
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        output_chars = set(output_text.upper())
        invalid_chars = output_chars - valid_chars
        
        if invalid_chars:
            return 0.3
            
        # Проверяем соотношение длин
        length_ratio = min(len(output_text), len(input_text)) / max(len(output_text), len(input_text))
        
        # Базовая уверенность на основе длины
        confidence = length_ratio * 0.8 + 0.2
        
        return min(1.0, confidence)

if __name__ == "__main__":
    # Пример использования
    model_path = "fine_tuned_model_multi"
    processor = CipherProcessor(model_path)
    
    # Тест шифрования
    plaintext = "HELLO WORLD"
    print(f"\nTesting encryption:")
    print(f"Plaintext: {plaintext}")
    
    for cipher in ['vigenere', 'substitution']:
        encrypted, conf = processor.encrypt(plaintext, cipher)
        print(f"\n{cipher.capitalize()} cipher:")
        print(f"Encrypted: {encrypted}")
        print(f"Confidence: {conf:.2f}")
        
        # Тест дешифрования
        decrypted, conf = processor.decrypt(encrypted, cipher)
        print(f"Decrypted: {decrypted}")
        print(f"Confidence: {conf:.2f}")