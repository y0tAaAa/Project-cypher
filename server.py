import os
import sys
import psycopg2
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from datetime import datetime
from psycopg2 import IntegrityError
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ─── 1) .env + UTF-8 ───────────────────────────────────────────────
load_dotenv()
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ─── 2) Инициализация Flask ────────────────────────────────────────
app = Flask(__name__)
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
    cur.execute("SELECT user_id, username, email FROM Users WHERE user_id=%s", (user_id,))
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
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url, sslmode="require")
    # По умолчанию, для локальной разработки
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "llm"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )

# ─── 7) Инициализация схемы ────────────────────────────────────────
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(256) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # … остальные таблицы как у вас …
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ─── 8) Decryptor (локальная HF-модель) ─────────────────────────────
MODEL_ID = "y0ta/fine_tuned_model"

class Decryptor:
    model = None
    tokenizer = None

    @staticmethod
    def load_model():
        if Decryptor.model is None:
            Decryptor.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            Decryptor.model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
            Decryptor.model.eval()

    @staticmethod
    def decrypt(ciphertext: str) -> str:
        Decryptor.load_model()
        prompt = f"Ciphertext: {ciphertext}\nPlaintext:"
        inputs = Decryptor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        # Генерим на CPU
        outputs = Decryptor.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        text = Decryptor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Обрезаем всё до Plaintext:
        return text.split("Plaintext:")[-1].strip()

decryptor = Decryptor()

# ─── 9) Маршруты ───────────────────────────────────────────────────
@app.route("/")
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
        cur.execute(
            "SELECT user_id, username, email, password_hash FROM Users WHERE email=%s",
            (email,)
        )
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and check_password_hash(user[3], pwd):
            login_user(User(user[0], user[1], user[2]))
            return jsonify({"message": "Успешный вход"}), 200
        return jsonify({"error": "Неверные данные"}), 401
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        conn = None
        cur = None
        try:
            data = request.get_json(force=True)
            username, email, pwd = data.get("username"), data.get("email"), data.get("password")
            if not (username and email and pwd):
                return jsonify({"error": "Все поля обязательны"}), 400
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO Users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, generate_password_hash(pwd))
            )
            conn.commit()
            return jsonify({"message": "Регистрация успешна"}), 201
        except IntegrityError:
            conn.rollback()
            return jsonify({"error": "Пользователь существует"}), 400
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route('/data')
@login_required
def data():
    # Логика для отображения данных
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
    data = request.get_json(force=True)
    ct = data.get("ciphertext")
    if not ct:
        return jsonify({"error": "Нет текста"}), 400
    dec = decryptor.decrypt(ct)
    return jsonify({"ciphertext": ct, "decrypted_text": dec})

# ─── 10) Запуск ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)