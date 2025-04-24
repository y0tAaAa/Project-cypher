import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import psycopg2
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import requests  # Added for manual user info fetching

# Загружаем переменные окружения из .env
load_dotenv()

# Устанавливаем кодировку UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Получаем абсолютные пути к папкам templates и static
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # Now the root directory (PROJECT-CYPHER/)
TEMPLATES_DIR = os.path.join(SRC_DIR, 'src', 'templates')  # PROJECT-CYPHER/src/templates/
STATIC_DIR = os.path.join(SRC_DIR, 'src', 'static')  # PROJECT-CYPHER/src/static/
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = 'your_secret_key_here'

# Настройка Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Настройка OAuth для Google
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

class User(UserMixin):
    def __init__(self, user_id, username, email):
        self.id = user_id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, username, email FROM Users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

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
decryptor = Decryptor(model_path="y0ta/fine_tuned_model", cipher_type="Caesar")

# Подключение к базе данных
def get_db_connection():
    database_url = os.getenv('DATABASE_URL', 'dbname=llm user=postgres password=Vlad222 host=localhost port=5432')
    return psycopg2.connect(database_url)

# Главная страница
@app.route('/')
@login_required
def index():
    print("Accessing main page (/)")
    return render_template('index.html')

# Страница входа
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, email, password_hash FROM Users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user[3], password):
            user_obj = User(user[0], user[1], user[2])
            login_user(user_obj)
            # Перенаправляем на главную страницу, игнорируя параметр next
            return jsonify({"message": "Login successful", "redirect": url_for('index')})
        else:
            return jsonify({"error": "Invalid email or password"}), 401

    return render_template('login.html')

# Маршрут для авторизации через Google
@app.route('/google/login')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

# Callback для Google OAuth
@app.route('/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        print("Token:", token)

        # Manually fetch user info using the access token
        userinfo_endpoint = 'https://www.googleapis.com/oauth2/v3/userinfo'
        headers = {'Authorization': f"Bearer {token['access_token']}"}
        response = requests.get(userinfo_endpoint, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        user_info = response.json()
        print("User Info:", user_info)

        email = user_info['email']
        username = user_info.get('name', email.split('@')[0])

        # Проверяем, есть ли пользователь в базе данных
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, email FROM Users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            # Пользователь уже существует, логиним его
            user_obj = User(user[0], user[1], user[2])
        else:
            # Создаем нового пользователя (без пароля, так как используется Google)
            cursor.execute(
                "INSERT INTO Users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING user_id",
                (username, email, '')
            )
            user_id = cursor.fetchone()[0]
            conn.commit()
            user_obj = User(user_id, username, email)

        cursor.close()
        conn.close()

        login_user(user_obj)
        # Перенаправляем на главную страницу, игнорируя параметр next
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error in Google callback: {str(e)}")
        return str(e), 500

# Страница регистрации
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO Users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING user_id",
                (username, email, password_hash)
            )
            user_id = cursor.fetchone()[0]
            conn.commit()
            return jsonify({"message": "Registration successful"})
        except psycopg2.IntegrityError as e:
            conn.rollback()
            return jsonify({"error": "Username or email already exists"}), 400
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')

# Выход
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login', success='logout'))

# Страница табличных данных
@app.route('/data')
@login_required
def data():
    return render_template('data.html')

# Страница истории
@app.route('/history')
@login_required
def history():
    return render_template('history.html')

# Страница настроек
@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

# Маршрут для получения попыток дешифровки
@app.route('/attempts')
@login_required
def get_attempts():
    sort_by = request.args.get('sort', 'start_time')
    order = request.args.get('order', 'asc')
    cipher_type = request.args.get('cipher_type')

    query = """
        SELECT 
            da.attempt_id, 
            c.name AS cipher_name, 
            m.name AS model_name, 
            da.start_time, 
            da.success,
            c.encrypted_text,
            dr.model_output
        FROM Decryption_Attempts da
        JOIN Cipher c ON da.cipher_id = c.cipher_id
        JOIN Model m ON da.model_id = m.model_id
        LEFT JOIN Decryption_Result dr ON c.cipher_id = dr.cipher_id
        WHERE da.user_id = %s
    """
    params = [current_user.id]

    if cipher_type:
        query += " AND c.cipher_type = %s"
        params.append(cipher_type)

    query += f" ORDER BY {sort_by} {'ASC' if order == 'asc' else 'DESC'}"

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    attempts = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify([
        {
            "attempt_id": attempt[0],
            "cipher_name": attempt[1],
            "model_name": attempt[2],
            "start_time": attempt[3].isoformat(),
            "success": attempt[4],
            "encrypted_text": attempt[5],
            "decrypted_text": attempt[6]
        }
        for attempt in attempts
    ])

# Маршрут для дешифровки
@app.route('/decrypt', methods=['POST'])
@login_required
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
        # 1. Создаем новую запись в таблице Cipher для каждой попытки
        cursor.execute("""
            INSERT INTO Cipher (name, historical_period, origin, encryption_principles, encrypted_text, discovery_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING cipher_id
        """, ("Caesar Cipher", "Ancient", "Rome", "Shift by 3", ciphertext, datetime.now().date()))
        cipher_id = cursor.fetchone()[0]

        # 2. Проверяем, есть ли модель в таблице Model
        cursor.execute("SELECT model_id FROM Model WHERE name = %s AND version = %s", ("GPT-2", "1.0"))
        model_result = cursor.fetchone()
        if model_result:
            model_id = model_result[0]
        else:
            cursor.execute("""
                INSERT INTO Model (name, specialization, version, usage_date)
                VALUES (%s, %s, %s, %s)
                RETURNING model_id
            """, ("GPT-2", "Decryption", "1.0", datetime.now().date()))
            model_id = cursor.fetchone()[0]

        # 3. Регистрируем попытку расшифровки
        cursor.execute("""
            INSERT INTO Decryption_Attempts (cipher_id, model_id, user_id, start_time, end_time, success, percent_correct)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING attempt_id
        """, (cipher_id, model_id, current_user.id, start_time, end_time, True, 0.0))
        attempt_id = cursor.fetchone()[0]

        # 4. Сохраняем результат расшифровки
        cursor.execute("""
            INSERT INTO Decryption_Result (cipher_id, model_output, similarity, readability)
            VALUES (%s, %s, %s, %s)
        """, (cipher_id, decrypted, 0.0, 0.0))

        conn.commit()

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({"ciphertext": ciphertext, "decrypted_text": decrypted})

# Инициализация базы данных
if __name__ == "__main__":
    conn = get_db_connection()
    cursor = conn.cursor()

    # Создание таблицы Users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(256) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Создание остальных таблиц (без удаления)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Cipher (
            cipher_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            cipher_type VARCHAR(50),
            historical_period VARCHAR(100),
            origin VARCHAR(100),
            encryption_principles TEXT,
            encrypted_text TEXT,
            discovery_date DATE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Model (
            model_id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            specialization VARCHAR(100),
            version VARCHAR(20),
            usage_date DATE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Decryption_Attempts (
            attempt_id SERIAL PRIMARY KEY,
            cipher_id INT,
            model_id INT,
            user_id INT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            success BOOLEAN,
            percent_correct DECIMAL(5,2),
            FOREIGN KEY (cipher_id) REFERENCES Cipher(cipher_id),
            FOREIGN KEY (model_id) REFERENCES Model(model_id),
            FOREIGN KEY (user_id) REFERENCES Users(user_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Decryption_Result (
            result_id SERIAL PRIMARY KEY,
            cipher_id INT,
            model_output TEXT,
            similarity DECIMAL(5,2),
            readability DECIMAL(5,2),
            FOREIGN KEY (cipher_id) REFERENCES Cipher(cipher_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Manual_Correction (
            correction_id SERIAL PRIMARY KEY,
            result_id INT,
            corrected_by VARCHAR(100),
            percent_changed DECIMAL(5,2),
            final_text TEXT,
            FOREIGN KEY (result_id) REFERENCES Decryption_Result(result_id)
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

    app.run(host='0.0.0.0', port=5000, debug=True)