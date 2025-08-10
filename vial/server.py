import hashlib
import uuid
import logging
import sqlite3
import os
from flask import Flask, request, jsonify
from vial_manager import VialManager
from datetime import datetime

app = Flask(__name__)
vial_manager = VialManager()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize SQLite database
db_path = '/app/vial.db'
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE vials (
        id TEXT PRIMARY KEY,
        status TEXT,
        code TEXT,
        codeLength INTEGER,
        isPython BOOLEAN,
        webxosHash TEXT,
        walletAddress TEXT,
        walletBalance REAL,
        tasks TEXT
    )''')
    c.execute('''CREATE TABLE wallets (
        address TEXT PRIMARY KEY,
        balance REAL
    )''')
    c.execute('''CREATE TABLE networks (
        networkId TEXT PRIMARY KEY
    )''')
    conn.commit()
    conn.close()

@app.route('/mcp/ping', methods=['GET'])
def ping():
    try:
        logger.info("Ping received")
        return jsonify({"status": "online"}), 200
    except Exception as e:
        logger.error(f"Ping error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/mcp/auth', methods=['POST'])
def auth():
    try:
        data = request.get_json()
        if not data or 'networkId' not in data:
            logger.error("Invalid auth request")
            return jsonify({"error": "Invalid request"}), 400
        token = str(uuid.uuid4())
        address = hashlib.sha256(token.encode()).hexdigest()
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO networks (networkId) VALUES (?)", (data['networkId'],))
        c.execute("INSERT OR REPLACE INTO wallets (address, balance) VALUES (?, ?)", (address, 0.0))
        conn.commit()
        conn.close()
        logger.info(f"Authenticated client with network ID: {data['networkId']}")
        return jsonify({"token": token, "address": address}), 200
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/mcp/train', methods=['POST'])
def train():
    try:
        if not request.form.get('networkId'):
            logger.error("Missing networkId in train request")
            return jsonify({"error": "Missing networkId"}), 400
        code = request.form.get('code')
        is_python = request.form.get('isPython') == 'true'
        network_id = request.form.get('networkId')
        if not code:
            logger.error("No code provided for training")
            return jsonify({"error": "No code provided"}), 400
        start_time = datetime.now()
        vials = vial_manager.train_vials(code, is_python, network_id)
        training_time = (datetime.now() - start_time).total_seconds()
        balance = 0.0004  # 0.0001 per vial
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        for vial in vials:
            tasks = ','.join(vial['tasks'])
            c.execute("INSERT OR REPLACE INTO vials (id, status, code, codeLength, isPython, webxosHash, walletAddress, walletBalance, tasks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (vial['id'], vial['status'], vial['code'], vial['codeLength'], vial['isPython'], vial['webxosHash'], vial['wallet']['address'], vial['wallet']['balance'], tasks))
        c.execute("UPDATE wallets SET balance = balance + ? WHERE address IN (SELECT walletAddress FROM vials WHERE id LIKE 'vial%')", (balance,))
        conn.commit()
        conn.close()
        logger.info(f"Training completed for network ID: {network_id}, earned {balance:.4f} $WEBXOS")
        return jsonify({"vials": vials, "balance": balance}), 200
    except Exception as e:
        logger.error(f"Train error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/mcp/void', methods=['POST'])
def void():
    try:
        data = request.get_json()
        if not data or 'networkId' not in data:
            logger.error("Invalid void request")
            return jsonify({"error": "Invalid request"}), 400
        vial_manager.void_vials(data['networkId'])
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("DELETE FROM vials WHERE id LIKE 'vial%'")
        c.execute("DELETE FROM wallets WHERE address IN (SELECT walletAddress FROM vials)")
        c.execute("DELETE FROM networks WHERE networkId = ?", (data['networkId'],))
        conn.commit()
        conn.close()
        logger.info(f"Voided vials for network ID: {data['networkId']}")
        return jsonify({"status": "voided"}), 200
    except Exception as e:
        logger.error(f"Void error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/mcp/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files or not request.form.get('networkId'):
            logger.error("Invalid upload request")
            return jsonify({"error": "Missing file or networkId"}), 400
        file = request.files['file']
        if not file.filename.endswith(('.py', '.js', '.txt', '.md')):
            logger.error("Invalid file type")
            return jsonify({"error": "Invalid file type"}), 400
        if file.content_length > 1024 * 1024:
            logger.error("File size exceeds 1MB")
            return jsonify({"error": "File size exceeds 1MB"}), 400
        file_path = f"/uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        logger.info(f"File uploaded to {file_path}")
        return jsonify({"filePath": file_path}), 200
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
