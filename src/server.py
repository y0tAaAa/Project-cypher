import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import psycopg2
from datetime import datetime

# Устанавливаем кодировку UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

app = Flask(__name__)

class Decryptor:
    def __init__(self, model_path, cipher_type):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cipher_type = cipher_type

    def decrypt(self, ciphertext):
        input_text = f"Ciphertext: {ciphertext}\nPlaintext:"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        decrypted = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Plaintext:" in decrypted:
            decrypted = decrypted.split("Plaintext:")[1].strip()
        return decrypted

# Инициализация дешифратора
model_path = "fine_tuned_model"
print(f"Loading model from: {model_path}")
decryptor = Decryptor(model_path=model_path, cipher_type="Caesar")

# Подключение к базе данных
def get_db_connection():
    return psycopg2.connect(
        "dbname=llm user=postgres password=Vlad222 host=localhost port=5432"
    )

@app.route('/decrypt', methods=['POST'])
def decrypt_text():
    data = request.get_json()
    ciphertext = data.get('ciphertext')
    if not ciphertext:
        return jsonify({"error": "No ciphertext provided"}), 400

    # Расшифровываем текст
    start_time = datetime.now()
    decrypted = decryptor.decrypt(ciphertext)
    end_time = datetime.now()

    # Подключаемся к базе данных
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Проверяем, есть ли шифр с name = "Caesar Cipher"
        cursor.execute("SELECT cipher_id FROM Ciphers WHERE name = %s", ("Caesar Cipher",))
        cipher_result = cursor.fetchone()
        if cipher_result:
            cipher_id = cipher_result[0]
        else:
            # Если шифра нет, добавляем его
            cursor.execute("""
                INSERT INTO Ciphers (name, historical_period, origin, encryption_principles, encrypted_text, discovery_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING cipher_id
            """, ("Caesar Cipher", "Ancient", "Rome", "Shift by 3", ciphertext, datetime.now().date()))
            cipher_id = cursor.fetchone()[0]

        # 2. Проверяем, есть ли модель в таблице Models
        cursor.execute("SELECT model_id FROM Models WHERE name = %s AND version = %s", ("GPT-2", "1.0"))
        model_result = cursor.fetchone()
        if model_result:
            model_id = model_result[0]
        else:
            # Если модели нет, добавляем её
            cursor.execute("""
                INSERT INTO Models (name, specialization, version)
                VALUES (%s, %s, %s)
                RETURNING model_id
            """, ("GPT-2", "Decryption", "1.0"))
            model_id = cursor.fetchone()[0]

        # 3. Регистрируем попытку расшифровки
        cursor.execute("""
            INSERT INTO Decryption_Attempts (cipher_id, model_id, start_time, end_time, success, correctness_percentage)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING attempt_id
        """, (cipher_id, model_id, start_time, end_time, True, 0.0))  # correctness_percentage пока 0
        attempt_id = cursor.fetchone()[0]

        # 4. Сохраняем результат расшифровки
        cursor.execute("""
            INSERT INTO Decryption_Results (attempt_id, model_output, similarity_measure, readability_level)
            VALUES (%s, %s, %s, %s)
        """, (attempt_id, decrypted, 0.0, 0.0))  # similarity_measure и readability_level пока 0

        conn.commit()

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({"ciphertext": ciphertext, "decrypted_text": decrypted})

if __name__ == "__main__":
    # Проверяем, что таблицы существуют (можно закомментировать после первого запуска)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Ciphers (
            cipher_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            historical_period VARCHAR(50),
            origin VARCHAR(50),
            encryption_principles TEXT,
            encrypted_text TEXT,
            discovery_date DATE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Models (
            model_id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            specialization VARCHAR(50),
            version VARCHAR(20),
            CONSTRAINT unique_model UNIQUE (name, version)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Decryption_Attempts (
            attempt_id SERIAL PRIMARY KEY,
            cipher_id INT NOT NULL,
            model_id INT NOT NULL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            success BOOLEAN,
            correctness_percentage DECIMAL(5,2) CHECK (correctness_percentage BETWEEN 0 AND 100),
            FOREIGN KEY (cipher_id) REFERENCES Ciphers(cipher_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Decryption_Results (
            result_id SERIAL PRIMARY KEY,
            attempt_id INT NOT NULL,
            model_output TEXT,
            similarity_measure DECIMAL(5,2) CHECK (similarity_measure BETWEEN 0 AND 100),
            readability_level DECIMAL(5,2) CHECK (readability_level BETWEEN 0 AND 100),
            FOREIGN KEY (attempt_id) REFERENCES Decryption_Attempts(attempt_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Manual_Corrections (
            correction_id SERIAL PRIMARY KEY,
            result_id INT NOT NULL,
            corrector VARCHAR(100),
            changed_percentage DECIMAL(5,2) CHECK (changed_percentage BETWEEN 0 AND 100),
            final_text TEXT,
            FOREIGN KEY (result_id) REFERENCES Decryption_Results(result_id)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

    app.run(host='0.0.0.0', port=5000, debug=True)