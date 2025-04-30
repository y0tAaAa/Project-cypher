import os
import sys
import psycopg2
import logging
import re
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from datetime import datetime
from psycopg2 import IntegrityError
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import difflib
import psutil

# ─── 1) .env + UTF-8 ───────────────────────────────────────────────
load_dotenv()
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ─── 2) Инициализация Flask ────────────────────────────────────────
app = Flask(__name__, template_folder="src/templates", static_folder="src/static")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(24).hex())
app.config["JSON_AS_ASCII"] = False

# ─── 3) Логирование ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ─── 4) Flask-Login ────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, user_id, username, email):
        self.id = user_id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT user_id, username, email FROM "Users" WHERE user_id=%s', (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return User(*row) if row else None

# ─── 5) OAuth (Google) ────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    access_token_url="https://accounts.google.com/o/oauth2/token",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
    userinfo_endpoint="https://www.googleapis.com/oauth2/v3/userinfo",
    client_kwargs={"scope": "openid email profile"}
)

# ─── 6) Подключение к Postgres ────────────────────────────────────
def get_db_connection():
    try:
        url = os.getenv("DATABASE_URL")
        if url:
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            conn = psycopg2.connect(url, sslmode="require")
            logging.info("Successfully connected to database via DATABASE_URL")
            return conn
        # По умолчанию, для локальной разработки
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "llm"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
        )
        logging.info("Successfully connected to local database")
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {str(e)}")
        raise

# ─── 7) Инициализация схемы ────────────────────────────────────────
def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Create Users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS "Users" (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logging.info("Table 'Users' created or already exists")

        # Create Ciphers table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS "Ciphers" (
                cipher_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                historical_period VARCHAR(50),
                origin VARCHAR(50),
                encryption_principles TEXT,
                encrypted_text TEXT,
                plaintext TEXT,
                discovery_date DATE
            )
        """)
        logging.info("Table 'Ciphers' created or already exists")

        # Create Models table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS "Models" (
                model_id SERIAL PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                specialization VARCHAR(50),
                version VARCHAR(20),
                CONSTRAINT unique_model UNIQUE (name, version)
            )
        """)
        logging.info("Table 'Models' created or already exists")

        # Create Decryption_Attempts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS "Decryption_Attempts" (
                attempt_id SERIAL PRIMARY KEY,
                cipher_id INT NOT NULL,
                model_id INT NOT NULL,
                user_id INT NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                success BOOLEAN,
                correctness_percentage DECIMAL(5,2) CHECK (correctness_percentage BETWEEN 0 AND 100),
                encrypted_text TEXT,
                decrypted_text TEXT,
                FOREIGN KEY (cipher_id) REFERENCES "Ciphers"(cipher_id),
                FOREIGN KEY (model_id) REFERENCES "Models"(model_id),
                FOREIGN KEY (user_id) REFERENCES "Users"(user_id)
            )
        """)
        logging.info("Table 'Decryption_Attempts' created or already exists")

        # Create Decryption_Results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS "Decryption_Results" (
                result_id SERIAL PRIMARY KEY,
                attempt_id INT NOT NULL,
                model_output TEXT,
                similarity_measure DECIMAL(5,2) CHECK (similarity_measure BETWEEN 0 AND 100),
                readability_level DECIMAL(5,2) CHECK (readability_level BETWEEN 0 AND 100),
                FOREIGN KEY (attempt_id) REFERENCES "Decryption_Attempts"(attempt_id)
            )
        """)
        logging.info("Table 'Decryption_Results' created or already exists")

        # Create Manual_Corrections table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS "Manual_Corrections" (
                correction_id SERIAL PRIMARY KEY,
                result_id INT NOT NULL,
                corrector VARCHAR(100),
                changed_percentage DECIMAL(5,2) CHECK (changed_percentage BETWEEN 0 AND 100),
                final_text TEXT,
                FOREIGN KEY (result_id) REFERENCES "Decryption_Results"(result_id)
            )
        """)
        logging.info("Table 'Manual_Corrections' created or already exists")

        conn.commit()
        cur.close()
        conn.close()
        logging.info("Database initialization completed successfully")
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}")
        raise

# ─── 8) Начальная загрузка данных ─────────────────────────────────
def seed_initial_data():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Начальный пользователь
        users = [
            ('testuser', 'testuser@example.com', 'testpassword')
        ]
        for username, email, password in users:
            password_hash = generate_password_hash(password)
            cur.execute("""
                INSERT INTO "Users" (username, email, password_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (email) DO NOTHING
            """, (username, email, password_hash))
            logging.info(f"Seeded user: {username}, {email}")

        # Начальные модели
        models = [
            ('DecryptoBot', 'Decryption', '1.0'),
            ('CipherMaster', 'Decryption', '2.0'),
            ('CryptoAI', 'Decryption', '1.2'),
        ]
        for name, spec, ver in models:
            cur.execute("""
                INSERT INTO "Models" (name, specialization, version)
                VALUES (%s, %s, %s)
                ON CONFLICT (name, version) DO NOTHING
            """, (name, spec, ver))
            logging.info(f"Seeded model: {name}, version: {ver}")

        # Начальные шифры с зашифрованным и эталонным текстом
        ciphers = [
            ('Caesar Cipher', 'Ancient', 'Rome', 'Shift by 3', 'Wklv lv d vhfuhw phvvdjh+', 'THIS IS A SECRET MESSAGE', '2025-04-26'),
            ('Vigenere Cipher', 'Medieval', 'France', 'Polyalphabetic', 'KHOOR ZRUOG', 'HELLO WORLD', '2025-04-26'),
            ('Enigma', 'WWII', 'Germany', 'Rotor machine', 'BJT QF UFJHLTK', 'THE ENEMY KNOWS', '2025-04-26'),
        ]
        for name, period, origin, princ, enc_text, plain_text, disc_date in ciphers:
            cur.execute("""
                INSERT INTO "Ciphers" (name, historical_period, origin, encryption_principles, encrypted_text, plaintext, discovery_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (name, period, origin, princ, enc_text, plain_text, disc_date))
            logging.info(f"Seeded cipher: {name}")

        conn.commit()
        cur.close()
        conn.close()
        logging.info("Initial data seeded successfully")
    except Exception as e:
        logging.error(f"Failed to seed initial data: {str(e)}")
        raise

# Вызов функций инициализации
init_db()
seed_initial_data()

# ─── 9) Decryptor (локальная HF-модель) ─────────────────────────────
MODEL_ID = "y0ta/fine_tuned_model"

class Decryptor:
    model = None
    tokenizer = None

    @staticmethod
    def load_model():
        if Decryptor.model is None:
            logging.info(f"Memory usage before model load: {psutil.virtual_memory().percent}%")
            Decryptor.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            Decryptor.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto"
            )
            Decryptor.model.eval()
            logging.info(f"Model loaded: {MODEL_ID}")
            logging.info(f"Memory usage after model load: {psutil.virtual_memory().percent}%")

    @staticmethod
    def decrypt(ciphertext: str) -> str:
        try:
            Decryptor.load_model()
            prompt = f"Ciphertext: {ciphertext}\nPlaintext:"
            inputs = Decryptor.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            outputs = Decryptor.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                num_beams=2,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            text = Decryptor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = text.split("Plaintext:")[-1].strip()
            logging.info(f"Decrypted text: {result}")
            return result
        except Exception as e:
            logging.error(f"Decryption error: {str(e)}")
            raise

decryptor = Decryptor()

# ─── 10) Маршруты ───────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json(force=True)
        email, pwd = data.get("email"), data.get("password")
        if not (email and pwd):
            return jsonify({"error": "Email и пароль обязательны"}), 400
        conn = get_db_connection()
        cur = conn.cursor()
        logging.info(f"Attempting to login with email: {email}")
        cur.execute(
            'SELECT user_id, username, email, password_hash FROM "Users" WHERE email=%s',
            (email,)
        )
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and check_password_hash(user[3], pwd):
            login_user(User(user[0], user[1], user[2]))
            logging.info(f"User logged in: {email}")
            return jsonify({"message": "Успешный вход", "redirect": "/"})
        logging.warning(f"Failed login attempt for email: {email}")
        return jsonify({"error": "Неверные данные"}), 401
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        conn = None
        cur = None
        try:
            data = request.get_json(force=True)
            username = data.get("username")
            email = data.get("email")
            password = data.get("password")
            if not (username and email and password):
                return jsonify({"error": "Все поля обязательны: username, email, password"}), 400
            if len(password) < 6:
                return jsonify({"error": "Пароль должен содержать минимум 6 символов"}), 400

            conn = get_db_connection()
            cur = conn.cursor()
            logging.info(f"Attempting to register user: {username}, {email}")
            pw_hash = generate_password_hash(password)
            cur.execute(
                'INSERT INTO "Users" (username, email, password_hash) VALUES (%s, %s, %s) RETURNING user_id',
                (username, email, pw_hash)
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            logging.info(f"User registered successfully: user_id={user_id}, username={username}, email={email}")
            return jsonify({"message": "Регистрация успешна! Пожалуйста, войдите."})
        except IntegrityError as e:
            logging.warning(f"Registration failed: IntegrityError - {str(e)}")
            return jsonify({"error": "Пользователь или email уже существует"}), 400
        except Exception as e:
            logging.error(f"Registration error: {str(e)}", exc_info=True)
            return jsonify({"error": f"Ошибка сервера: {str(e)}"}), 500
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logging.info(f"User logged out: {current_user.email}")
    logout_user()
    return redirect(url_for("index"))

@app.route('/data')
@login_required
def data():
    return render_template('data.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route("/decrypt", methods=["POST"])
@login_required
def decrypt_text():
    try:
        # Получение данных из запроса
        data = request.get_json(force=True)
        ct = data.get("ciphertext")
        cipher_id = data.get("cipher_id")
        model_id = data.get("model_id")

        # Валидация входных данных
        if not ct:
            return jsonify({"error": "Zašifrovaný text je povinný"}), 400
        if not cipher_id or not model_id:
            return jsonify({"error": "Nie je uvedené cipher_id alebo model_id"}), 400
        try:
            cipher_id = int(cipher_id)
            model_id = int(model_id)
        except (ValueError, TypeError):
            return jsonify({"error": "cipher_id a model_id musia byť celé čísla"}), 400

        # Подключение к базе данных
        conn = get_db_connection()
        cur = conn.cursor()

        # Проверка существования пользователя
        cur.execute('SELECT user_id FROM "Users" WHERE user_id = %s', (current_user.id,))
        user_exists = cur.fetchone()
        if not user_exists:
            cur.close()
            conn.close()
            logout_user()
            return jsonify({"error": "Používateľ nebol nájdený. Prosím, prihláste sa znova."}), 401

        # Проверка существования шифра
        cur.execute('SELECT plaintext FROM "Ciphers" WHERE cipher_id = %s', (cipher_id,))
        cipher_data = cur.fetchone()
        if not cipher_data:
            cur.close()
            conn.close()
            return jsonify({"error": "Šifra nebola nájdená"}), 404
        reference_text = cipher_data[0]  # Эталонный текст

        # Проверка существования модели
        cur.execute('SELECT model_id FROM "Models" WHERE model_id = %s', (model_id,))
        model_exists = cur.fetchone()
        if not model_exists:
            cur.close()
            conn.close()
            return jsonify({"error": "Model nebol nájdený"}), 404

        # Дешифровка
        start_time = datetime.now()
        try:
            dec = decryptor.decrypt(ct)
        except Exception as e:
            logging.error(f"Decryption failed for ciphertext '{ct}': {str(e)}")
            cur.close()
            conn.close()
            return jsonify({"error": f"Chyba pri dešifrovaní: {str(e)}"}), 500

        end_time = datetime.now()
        success = bool(re.search(r'\b\w+\b', dec))
        correctness_percentage = 100.0 if success else 0.0

        # Вставка в Decryption_Attempts
        cur.execute("""
            INSERT INTO "Decryption_Attempts" 
            (cipher_id, model_id, user_id, start_time, end_time, success, correctness_percentage, encrypted_text, decrypted_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING attempt_id
        """, (cipher_id, model_id, current_user.id, start_time, end_time, success, correctness_percentage, ct, dec))
        attempt_id = cur.fetchone()[0]

        # Расчёт similarity_measure и readability_level
        if reference_text:
            seq_matcher = difflib.SequenceMatcher(None, dec.lower(), reference_text.lower())
            similarity_measure = round(seq_matcher.ratio() * 100, 2)
        else:
            similarity_measure = round(min(100.0, (len(dec) / len(ct)) * 100 if len(ct) > 0 else 0.0), 2)

        # readability_level: Процент слов, которые выглядят "читаемыми"
        words = dec.split()
        readable_words = sum(1 for word in words if word.isalpha())
        readability_level = round((readable_words / len(words) * 100) if words else 0.0, 2)

        # Вставка в Decryption_Results
        cur.execute("""
            INSERT INTO "Decryption_Results" 
            (attempt_id, model_output, similarity_measure, readability_level)
            VALUES (%s, %s, %s, %s)
            RETURNING result_id
        """, (attempt_id, dec, similarity_measure, readability_level))
        result_id = cur.fetchone()[0]

        # Коммит транзакции
        conn.commit()
        logging.info(f"Decryption attempt logged: attempt_id={attempt_id}, cipher_id={cipher_id}, model_id={model_id}, success={success}")
        logging.info(f"Decryption result logged: attempt_id={attempt_id}, result_id={result_id}, similarity_measure={similarity_measure}, readability_level={readability_level}")

        # Возвращаем результат
        return jsonify({
            "ciphertext": ct,
            "decrypted_text": dec,
            "reference_text": reference_text,
            "similarity_measure": similarity_measure,
            "readability_level": readability_level,
            "attempt_id": attempt_id,
            "result_id": result_id
        })

    except Exception as e:
        logging.error(f"Unexpected error in /decrypt: {str(e)}")
        return jsonify({"error": f"Chyba na serveri: {str(e)}"}), 500

    finally:
        # Гарантированное закрытие соединения
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

@app.route("/correct", methods=["POST"])
@login_required
def correct_decryption():
    data = request.get_json(force=True)
    result_id = data.get("result_id")
    final_text = data.get("final_text")
    if not result_id or not final_text:
        return jsonify({"error": "Не указаны result_id или final_text"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Проверка существования результата
        cur.execute('SELECT model_output FROM "Decryption_Results" WHERE result_id = %s', (result_id,))
        result = cur.fetchone()
        if not result:
            cur.close()
            conn.close()
            return jsonify({"error": "Результат не найден"}), 404

        original_text = result[0]
        # Расчёт changed_percentage с помощью SequenceMatcher
        seq_matcher = difflib.SequenceMatcher(None, original_text.lower(), final_text.lower())
        changed_percentage = (1 - seq_matcher.ratio()) * 100  # Процент изменений

        # Вставка в Manual_Corrections
        cur.execute("""
            INSERT INTO "Manual_Corrections" 
            (result_id, corrector, changed_percentage, final_text)
            VALUES (%s, %s, %s, %s)
        """, (result_id, current_user.username, changed_percentage, final_text))

        conn.commit()
        logging.info(f"Manual correction logged: result_id={result_id}, corrector={current_user.username}, changed_percentage={changed_percentage}")
        return jsonify({"message": "Исправление успешно сохранено", "changed_percentage": changed_percentage})
    except Exception as e:
        logging.error(f"Correction failed: {e}")
        return jsonify({"error": "Ошибка при сохранении исправления: " + str(e)}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/attempts')
@login_required
def attempts():
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT da.attempt_id, c.name AS cipher_name, m.name AS model_name,
                   da.start_time, da.end_time, da.success, da.correctness_percentage,
                   da.encrypted_text, da.decrypted_text,
                   dr.result_id, dr.model_output, dr.similarity_measure, dr.readability_level,
                   mc.correction_id, mc.corrector, mc.changed_percentage, mc.final_text
            FROM "Decryption_Attempts" da
            JOIN "Ciphers" c ON da.cipher_id = c.cipher_id
            JOIN "Models" m ON da.model_id = m.model_id
            LEFT JOIN "Decryption_Results" dr ON da.attempt_id = dr.attempt_id
            LEFT JOIN "Manual_Corrections" mc ON dr.result_id = mc.result_id
            WHERE da.user_id = %s
            ORDER BY da.start_time DESC
        """, (current_user.id,))
        attempts = cur.fetchall()
        attempts_list = [
            {
                "attempt_id": row[0],
                "cipher_name": row[1],
                "model_name": row[2],
                "start_time": row[3].isoformat() if row[3] else None,
                "end_time": row[4].isoformat() if row[4] else None,
                "success": row[5],
                "correctness_percentage": float(row[6]) if row[6] is not None else None,
                "encrypted_text": row[7],
                "decrypted_text": row[8],
                "result_id": row[9],
                "model_output": row[10],
                "similarity_measure": float(row[11]) if row[11] is not None else None,
                "readability_level": float(row[12]) if row[12] is not None else None,
                "correction_id": row[13],
                "corrector": row[14],
                "changed_percentage": float(row[15]) if row[15] is not None else None,
                "final_text": row[16]
            }
            for row in attempts
        ]
        logging.info(f"Fetched {len(attempts_list)} decryption attempts for user_id={current_user.id}")
        return jsonify(attempts_list)
    except Exception as e:
        logging.error(f"Error fetching attempts: {e}")
        return jsonify({"error": "Ошибка при получении данных"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

@app.route('/api/ciphers', methods=['GET'])
@login_required
def get_ciphers():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT cipher_id, name, encrypted_text, plaintext FROM "Ciphers"')
    ciphers = [{"cipher_id": row[0], "name": row[1], "encrypted_text": row[2], "plaintext": row[3]} for row in cur.fetchall()]
    cur.close()
    conn.close()
    logging.info(f"Fetched ciphers: {ciphers}")
    return jsonify(ciphers)

@app.route('/api/models', methods=['GET'])
@login_required
def get_models():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT model_id, name, specialization, version FROM "Models"')
        models = [{"model_id": row[0], "name": row[1], "specialization": row[2], "version": row[3]} for row in cur.fetchall()]
        cur.close()
        conn.close()
        logging.info(f"Fetched models: {models}")
        return jsonify(models)
    except Exception as e:
        logging.error(f"Error fetching models: {e}")
        return jsonify({"error": "Failed to fetch models"}), 500

@app.route('/update_password', methods=['POST'])
@login_required
def update_password():
    data = request.get_json(force=True)
    new_password = data.get('new_password')
    if not new_password:
        return jsonify({"error": "Новое пароль обязателен"}), 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        'UPDATE "Users" SET password_hash = %s WHERE user_id = %s',
        (generate_password_hash(new_password), current_user.id)
    )
    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"Password updated for user_id={current_user.id}")
    return jsonify({"message": "Heslo bolo úspešne aktualizované"}), 200

# Добавляем заголовок Content-Type для всех ответов
@app.after_request
def add_header(response):
    if response.mimetype == 'text/html':
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
    elif response.mimetype == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# ─── 11) Запуск ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)