-- Таблица Ciphers
CREATE TABLE IF NOT EXISTS Ciphers (
    cipher_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    historical_period VARCHAR(50),
    origin VARCHAR(50),
    encryption_principles TEXT,
    encrypted_text TEXT,
    discovery_date DATE
);

-- Таблица Models
CREATE TABLE IF NOT EXISTS Models (
    model_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    specialization VARCHAR(50),
    version VARCHAR(20),
    CONSTRAINT unique_model UNIQUE (name, version)
);

-- Таблица Decryption_Attempts
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
);

-- Таблица Decryption_Results
CREATE TABLE IF NOT EXISTS Decryption_Results (
    result_id SERIAL PRIMARY KEY,
    attempt_id INT NOT NULL,
    model_output TEXT,
    similarity_measure DECIMAL(5,2) CHECK (similarity_measure BETWEEN 0 AND 100),
    readability_level DECIMAL(5,2) CHECK (readability_level BETWEEN 0 AND 100),
    FOREIGN KEY (attempt_id) REFERENCES Decryption_Attempts(attempt_id)
);

-- Таблица Manual_Corrections
CREATE TABLE IF NOT EXISTS Manual_Corrections (
    correction_id SERIAL PRIMARY KEY,
    result_id INT NOT NULL,
    corrector VARCHAR(100),
    changed_percentage DECIMAL(5,2) CHECK (changed_percentage BETWEEN 0 AND 100),
    final_text TEXT,
    FOREIGN KEY (result_id) REFERENCES Decryption_Results(result_id)
);