import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional
import os

# Импорты для LLM
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class Decryptor:
    def __init__(self, model_path: Optional[str] = None, cipher_type: str = "Caesar"):
        """
        Инициализация дешифратора.
        :param model_path: Путь к дообученной модели (если используется LLM).
        :param cipher_type: Тип шифра (по умолчанию "Caesar").
        """
        self.model = None
        self.tokenizer = None
        self.cipher_type = cipher_type.lower()

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

    def decrypt(self, ciphertext: str, cipher_id: int, model_id: int, method: str = "analytical", max_length: int = 50) -> str:
        """
        Дешифрует текст и записывает результаты в базу данных.
        :param ciphertext: Зашифрованный текст.
        :param cipher_id: ID шифра в базе.
        :param model_id: ID модели в базе.
        :param method: Метод дешифрования ("analytical" или "llm").
        :param max_length: Максимальная длина вывода для LLM.
        :return: Расшифрованный текст.
        """
        conn = self._get_connection()
        start_time = datetime.now()

        # Выбор метода дешифрования
        if method == "analytical":
            plaintext = self._decrypt_analytical(ciphertext)
        elif method == "llm" and self.model and self.tokenizer:
            plaintext = self._decrypt_with_llm(ciphertext, max_length)
        else:
            raise ValueError("Invalid method or LLM not available")

        end_time = datetime.now()
        # Проверка успешности (для теста сравниваем с "hello world")
        success = plaintext.lower() == "hello world"
        correctness_percentage = 100.0 if success else self._calculate_correctness(ciphertext, plaintext)

        # Запись в базу данных
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Вставка попытки дешифрования
                cur.execute("""
                    INSERT INTO Decryption_Attempts (cipher_id, model_id, start_time, end_time, success, correctness_percentage)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING attempt_id;
                """, (cipher_id, model_id, start_time, end_time, success, correctness_percentage))
                attempt_id = cur.fetchone()["attempt_id"]

                # Вставка результата дешифрования
                cur.execute("""
                    INSERT INTO Decryption_Results (attempt_id, model_output, similarity_measure, readability_level)
                    VALUES (%s, %s, %s, %s);
                """, (attempt_id, plaintext, correctness_percentage, correctness_percentage))

                conn.commit()
        except Exception as e:
            print("Ошибка при записи в БД:", e)
            conn.rollback()
            raise
        finally:
            conn.close()

        return plaintext

    def _decrypt_analytical(self, ciphertext: str) -> str:
        """
        Аналитическое дешифрование в зависимости от типа шифра.
        :param ciphertext: Зашифрованный текст.
        :return: Расшифрованный текст.
        """
        if self.cipher_type == "caesar":
            return self._decrypt_caesar(ciphertext, shift=3)
        else:
            raise ValueError(f"Unsupported cipher type: {self.cipher_type}")

    def _decrypt_caesar(self, ciphertext: str, shift: int = 3) -> str:
        """
        Дешифрование шифра Цезаря с заданным сдвигом.
        :param ciphertext: Зашифрованный текст.
        :param shift: Сдвиг (по умолчанию 3).
        :return: Расшифрованный текст.
        """
        decrypted = ""
        for char in ciphertext:
            if char.isalpha():
                ascii_base = 65 if char.isupper() else 97
                decrypted_char = chr((ord(char) - ascii_base - shift) % 26 + ascii_base)
                decrypted += decrypted_char
            else:
                decrypted += char
        return decrypted

    def _decrypt_with_llm(self, ciphertext: str, max_length: int) -> str:
        """
        Дешифрование с использованием LLM.
        :param ciphertext: Зашифрованный текст.
        :param max_length: Максимальная длина вывода.
        :return: Расшифрованный текст.
        """
        prompt = f"Cipher: {self.cipher_type}\nCiphertext: {ciphertext}\nPlaintext:"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=False,
            num_return_sequences=1
        )
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        plaintext = decoded_output.split("Plaintext:")[-1].strip()
        return plaintext

    def _calculate_correctness(self, ciphertext: str, plaintext: str) -> float:
        """
        Вычисляет процент правильности дешифрования (упрощённая метрика).
        :param ciphertext: Зашифрованный текст.
        :param plaintext: Расшифрованный текст.
        :return: Процент правильности (0-100).
        """
        if len(ciphertext) != len(plaintext):
            return 0.0
        matches = sum(1 for c1, c2 in zip(ciphertext.lower(), plaintext.lower()) if c1 == c2)
        return (matches / len(ciphertext)) * 100

    def _get_connection(self):
        """
        Устанавливает соединение с базой данных.
        :return: Соединение с базой.
        """
        try:
            return psycopg2.connect(
                host="localhost",
                database="llm",
                user="y0ta",
                password="4572"
            )
        except Exception as e:
            print("Ошибка подключения к БД:", e)
            raise

if __name__ == "__main__":
    # Тест аналитического метода
    decryptor_analytical = Decryptor(cipher_type="Caesar")
    sample_ciphertext = "KHOOR ZRUOG"
    cipher_id = 1
    model_id = 1
    decrypted_text_analytical = decryptor_analytical.decrypt(sample_ciphertext, cipher_id, model_id, method="analytical")
    print(f"Analytical - Ciphertext: {sample_ciphertext}")
    print(f"Analytical - Decrypted Text: {decrypted_text_analytical}")

    # Тест LLM-метода с дообученной моделью
    model_path = os.path.join("..", "fine_tuned_model")  # Путь к модели относительно src/
    decryptor_llm = Decryptor(model_path=model_path, cipher_type="Caesar")
    decrypted_text_llm = decryptor_llm.decrypt(sample_ciphertext, cipher_id, model_id, method="llm")
    print(f"LLM - Ciphertext: {sample_ciphertext}")
    print(f"LLM - Decrypted Text: {decrypted_text_llm}")