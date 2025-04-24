import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from psycopg2 import IntegrityError
import psycopg2
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Ensure UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Paths for templates and static files
SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SRC_DIR, 'src', 'templates')
STATIC_DIR    = os.path.join(SRC_DIR, 'src', 'static')

# Initialize Flask
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret')

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# OAuth (Google) setup
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
    userinfo_endpoint='https://www.googleapis.com/oauth2/v3/userinfo',
    client_kwargs={'scope': 'openid email profile'},
)

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email):
        self.id = user_id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT user_id, username, email FROM Users WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    return User(*row) if row else None

# Database connection with fallback
def get_db_connection():
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(database_url, sslmode='require')
    return psycopg2.connect(
        dbname   = os.getenv('DB_NAME',     'llm'),
        user     = os.getenv('DB_USER',     'postgres'),
        password = os.getenv('DB_PASSWORD', 'Vlad222'),
        host     = os.getenv('DB_HOST',     'localhost'),
        port     = os.getenv('DB_PORT',     '5432'),
    )

# Create tables if they don't exist
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

# Run schema initialization immediately (Gunicorn will import this module)
init_db()

# Decryptor with 8-bit quantization
class Decryptor:
    def __init__(self, model_path="y0ta/fine_tuned_model", cipher_type="Caesar"):
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cipher_type = cipher_type

    def decrypt(self, ciphertext: str) -> str:
        prompt = f"Ciphertext: {ciphertext}\nPlaintext:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Plaintext:")[-1].strip()

# Instantiate decryptor
decryptor = Decryptor()

# Routes

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        data = request.get_json(force=True)
        email = data.get('email'); password = data.get('password')
        if not (email and password):
            return jsonify({"error": "Email and password are required"}), 400

        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(
            "SELECT user_id, username, email, password_hash FROM Users WHERE email = %s",
            (email,)
        )
        user = cur.fetchone()
        cur.close(); conn.close()

        if user and check_password_hash(user[3], password):
            login_user(User(user[0], user[1], user[2]))
            return jsonify({"message": "Login successful", "redirect": url_for('index')})
        return jsonify({"error": "Invalid email or password"}), 401

    return render_template('login.html')

@app.route('/google/login')
def google_login():
    return google.authorize_redirect(url_for('google_callback', _external=True))

@app.route('/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        headers = {'Authorization': f"Bearer {token['access_token']}"}
        resp = requests.get('https://www.googleapis.com/oauth2/v3/userinfo', headers=headers)
        resp.raise_for_status()
        info = resp.json()
        email = info['email']
        username = info.get('name', email.split('@')[0])

        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT user_id, username, email FROM Users WHERE email = %s", (email,))
        row = cur.fetchone()
        if row:
            user_obj = User(*row)
        else:
            cur.execute(
                "INSERT INTO Users (username, email, password_hash) VALUES (%s,%s,%s) RETURNING user_id",
                (username, email, '')
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            user_obj = User(user_id, username, email)
        cur.close(); conn.close()

        login_user(user_obj)
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error("Google callback error", exc_info=True)
        return str(e), 500

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        conn = None; cur = None
        try:
            data = request.get_json(force=True)
            username = data.get('username')
            email    = data.get('email')
            password = data.get('password')
            if not (username and email and password):
                return jsonify({"error": "Všetky polia sú povinné"}), 400

            conn = get_db_connection(); cur = conn.cursor()
            pw_hash = generate_password_hash(password)
            cur.execute(
                "INSERT INTO Users (username, email, password_hash) VALUES (%s,%s,%s)",
                (username, email, pw_hash)
            )
            conn.commit()
            return jsonify({"message": "Registrácia úspešná"})
        except IntegrityError:
            return jsonify({"error": "Užívateľ alebo email už existuje"}), 400
        except Exception as e:
            app.logger.error("Register error", exc_info=True)
            return jsonify({"error": str(e)}), 500
        finally:
            if cur: cur.close()
            if conn: conn.close()

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

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

@app.route('/attempts')
@login_required
def get_attempts():
    sort_by     = request.args.get('sort','start_time')
    order       = request.args.get('order','asc')
    cipher_type = request.args.get('cipher_type')

    query = """
        SELECT da.attempt_id, c.name, m.name, da.start_time, da.success,
               c.encrypted_text, dr.model_output
        FROM Decryption_Attempts da
        JOIN Cipher c ON da.cipher_id=c.cipher_id
        JOIN Model m ON da.model_id=m.model_id
        LEFT JOIN Decryption_Result dr ON c.cipher_id=dr.cipher_id
        WHERE da.user_id=%s
    """
    params = [current_user.id]
    if cipher_type:
        query += " AND c.cipher_type=%s"; params.append(cipher_type)
    query += f" ORDER BY {sort_by} {'ASC' if order=='asc' else 'DESC'}"

    conn = get_db_connection(); cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close(); conn.close()

    return jsonify([
        {
            "attempt_id": r[0], "cipher_name": r[1], "model_name": r[2],
            "start_time": r[3].isoformat(), "success": r[4],
            "encrypted_text": r[5], "decrypted_text": r[6]
        } for r in rows
    ])

@app.route('/decrypt', methods=['POST'])
@login_required
def decrypt_text():
    data = request.get_json(force=True)
    ciphertext = data.get('ciphertext')
    if not ciphertext:
        return jsonify({"error":"No ciphertext provided"}),400

    start     = datetime.now()
    decrypted = decryptor.decrypt(ciphertext)
    end       = datetime.now()

    conn = get_db_connection(); cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO Cipher (name,historical_period,origin,encryption_principles,
                                encrypted_text,discovery_date)
            VALUES (%s,%s,%s,%s,%s,%s) RETURNING cipher_id
        """, ("Caesar Cipher","Ancient","Rome","Shift by 3",ciphertext,start.date()))
        cid = cur.fetchone()[0]

        cur.execute("SELECT model_id FROM Model WHERE name=%s AND version=%s",
                    ("GPT-2","1.0"))
        mrow = cur.fetchone()
        if mrow:
            mid = mrow[0]
        else:
            cur.execute("""
                INSERT INTO Model (name,specialization,version,usage_date)
                VALUES (%s,%s,%s,%s) RETURNING model_id
            """, ("GPT-2","Decryption","1.0",start.date()))
            mid = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO Decryption_Attempts (cipher_id,model_id,user_id,
                start_time,end_time,success,percent_correct)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (cid,mid,current_user.id,start,end,True,0.0))

        cur.execute("""
            INSERT INTO Decryption_Result (cipher_id,model_output,similarity,readability)
            VALUES (%s,%s,%s,%s)
        """, (cid,decrypted,0.0,0.0))

        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"error":str(e)}),500
    finally:
        cur.close(); conn.close()

    return jsonify({"ciphertext":ciphertext,"decrypted_text":decrypted})

# Local development entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
