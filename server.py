import os
import sys
import psycopg2
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from datetime import datetime
from psycopg2 import IntegrityError

# ─── 1) Загрузка .env и установка UTF-8 ─────────────────────────────
load_dotenv()
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ─── 2) Пути к шаблонам и статикам ─────────────────────────────────
SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SRC_DIR, "src", "templates")
STATIC_DIR    = os.path.join(SRC_DIR, "src", "static")

# ─── 3) Инициализация Flask ────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.getenv("SECRET_KEY", "change-me")

# ─── 4) Проверка и нормализация HF_SPACE_URL ───────────────────────
HF_SPACE_URL = os.getenv("HF_SPACE_URL")
if not HF_SPACE_URL:
    raise RuntimeError("HF_SPACE_URL must be set")

# Переводим возможный /api/predict → /run/predict и убираем лишние слэши
HF_SPACE_URL = HF_SPACE_URL.rstrip("/") \
               .replace("/api/predict", "/run/predict")
app.logger.info(f"Using HF_SPACE_URL = {HF_SPACE_URL!r}")

# ─── 5) Flask-Login ────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, user_id, username, email):
        self.id       = user_id
        self.username = username
        self.email    = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute(
        "SELECT user_id, username, email FROM Users WHERE user_id = %s",
        (user_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return User(*row) if row else None

# ─── 6) OAuth (Google) ────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    access_token_url="https://accounts.google.com/o/oauth2/token",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
    userinfo_endpoint="https://www.googleapis.com/oauth2/v3/userinfo",
    client_kwargs={"scope": "openid email profile"},
)

# ─── 7) Подключение к PostgreSQL ──────────────────────────────────
def get_db_connection():
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url, sslmode="require")
    return psycopg2.connect(
        dbname   = os.getenv("DB_NAME",     "llm"),
        user     = os.getenv("DB_USER",     "postgres"),
        password = os.getenv("DB_PASSWORD", "password"),
        host     = os.getenv("DB_HOST",     "localhost"),
        port     = os.getenv("DB_PORT",     "5432"),
    )

# ─── 8) Инициализация схемы ────────────────────────────────────────
def init_db():
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Users (
      user_id SERIAL PRIMARY KEY,
      username VARCHAR(50) UNIQUE NOT NULL,
      email VARCHAR(100) UNIQUE NOT NULL,
      password_hash VARCHAR(256) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Cipher (
      cipher_id SERIAL PRIMARY KEY,
      name VARCHAR(100) NOT NULL,
      cipher_type VARCHAR(50),
      historical_period VARCHAR(100),
      origin VARCHAR(100),
      encryption_principles TEXT,
      encrypted_text TEXT,
      discovery_date DATE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Model (
      model_id SERIAL PRIMARY KEY,
      name VARCHAR(50) NOT NULL,
      specialization VARCHAR(100),
      version VARCHAR(20),
      usage_date DATE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Decryption_Attempts (
      attempt_id SERIAL PRIMARY KEY,
      cipher_id INT REFERENCES Cipher(cipher_id),
      model_id INT REFERENCES Model(model_id),
      user_id INT REFERENCES Users(user_id),
      start_time TIMESTAMP,
      end_time TIMESTAMP,
      success BOOLEAN,
      percent_correct DECIMAL(5,2)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Decryption_Result (
      result_id SERIAL PRIMARY KEY,
      cipher_id INT REFERENCES Cipher(cipher_id),
      model_output TEXT,
      similarity DECIMAL(5,2),
      readability DECIMAL(5,2)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Manual_Correction (
      correction_id SERIAL PRIMARY KEY,
      result_id INT REFERENCES Decryption_Result(result_id),
      corrected_by VARCHAR(100),
      percent_changed DECIMAL(5,2),
      final_text TEXT
    )""")
    conn.commit()
    cur.close()
    conn.close()

# выполняем один раз при импорте
init_db()

# ─── 9) Decryptor через HF Space ──────────────────────────────────
class Decryptor:
    def decrypt(self, ciphertext: str) -> str:
        payload = {
            "data": [ciphertext],
            "fn_index": 0
        }
        resp = requests.post(
            HF_SPACE_URL,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["data"][0]

decryptor = Decryptor()

# ─── 10) Health-check ─────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "hf_space_url": HF_SPACE_URL
    })

# ─── 11) Маршруты приложения ───────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        data     = request.get_json(force=True)
        email    = data.get("email")
        password = data.get("password")
        if not (email and password):
            return jsonify({"error": "Email and password are required"}), 400

        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT user_id, username, email, password_hash FROM Users WHERE email = %s",
            (email,)
        )
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user[3], password):
            login_user(User(user[0], user[1], user[2]))
            return jsonify({
                "message":  "Login successful",
                "redirect": url_for("index")
            })
        return jsonify({"error": "Invalid email or password"}), 401

    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        conn = None; cur = None
        try:
            data     = request.get_json(force=True)
            username = data.get("username")
            email    = data.get("email")
            password = data.get("password")
            if not (username and email and password):
                return jsonify({"error": "All fields required"}), 400

            conn    = get_db_connection()
            cur     = conn.cursor()
            pw_hash = generate_password_hash(password)
            cur.execute(
                "INSERT INTO Users (username,email,password_hash) VALUES (%s,%s,%s)",
                (username, email, pw_hash)
            )
            conn.commit()
            return jsonify({"message": "Registration successful"})
        except IntegrityError:
            return jsonify({"error": "User or email already exists"}), 400
        except Exception as e:
            app.logger.error("Register error", exc_info=True)
            return jsonify({"error": str(e)}), 500
        finally:
            if cur: cur.close()
            if conn: conn.close()

    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/decrypt", methods=["POST"])
@login_required
def decrypt_text():
    data       = request.get_json(force=True)
    ciphertext = data.get("ciphertext")
    if not ciphertext:
        return jsonify({"error": "No ciphertext provided"}), 400

    decrypted = decryptor.decrypt(ciphertext)
    return jsonify({
        "ciphertext":      ciphertext,
        "decrypted_text": decrypted
    })

# ─── 12) Запуск локально ───────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
