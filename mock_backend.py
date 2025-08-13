from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import hashlib
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(filename='/db/errorlog.md', level=logging.INFO, format='## [%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
VIAL_VERSION = '2.8'

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# In-memory storage
users = {}
wallets = {}
vials = {
    'vial1': {'status': 'stopped', 'tasks': [], 'config': {}},
    'vial2': {'status': 'stopped', 'tasks': [], 'config': {}},
    'vial3': {'status': 'stopped', 'tasks': [], 'config': {}},
    'vial4': {'status': 'stopped', 'tasks': [], 'config': {}}
}
blockchain = []

@app.route('/api/health', methods=['GET'])
def health():
    try:
        logger.info("Health check requested")
        return jsonify({"status": "healthy", "mongo": True, "version": VIAL_VERSION, "services": ["auth", "wallet", "vials"]})
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        if not user_id:
            logger.error("Login failed: Missing userId")
            return jsonify({"detail": "Missing userId"}), 400
        wallet_address = hashlib.sha256(user_id.encode()).hexdigest()
        wallet_hash = hashlib.sha256((user_id + str(datetime.utcnow())).encode()).hexdigest()
        api_key = f"JWT-{uuid.uuid4()}"
        users[user_id] = {"wallet_address": wallet_address, "wallet_hash": wallet_hash, "api_key": api_key}
        logger.info(f"User {user_id} logged in")
        return jsonify({"apiKey": api_key, "walletAddress": wallet_address, "walletHash": wallet_hash})
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"detail": f"Login failed: {str(e)}"}), 500

@app.route('/api/auth/api-key/generate', methods=['POST'])
def generate_api_key():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        if not user_id:
            logger.error("API key generation failed: Missing userId")
            return jsonify({"detail": "Missing userId"}), 400
        wallet_address = hashlib.sha256(user_id.encode()).hexdigest()
        wallet_hash = hashlib.sha256((user_id + str(datetime.utcnow())).encode()).hexdigest()
        api_key = f"JWT-{uuid.uuid4()}"
        users[user_id] = {"wallet_address": wallet_address, "wallet_hash": wallet_hash, "api_key": api_key}
        logger.info(f"Generated API key for user {user_id}")
        return jsonify({"apiKey": api_key, "walletAddress": wallet_address, "walletHash": wallet_hash})
    except Exception as e:
        logger.error(f"API key generation error: {str(e)}")
        return jsonify({"detail": f"API key generation failed: {str(e)}"}), 500

@app.route('/api/log_error', methods=['POST'])
def log_error():
    try:
        data = request.get_json()
        logger.error(f"Frontend error: {data['error']} at {data['endpoint']}")
        return jsonify({"status": "logged"})
    except Exception as e:
        logger.error(f"Error logging failed: {str(e)}")
        return jsonify({"detail": f"Error logging failed: {str(e)}"}), 500

@app.route('/api/vials/<vial_id>/prompt', methods=['POST'])
def send_prompt(vial_id):
    try:
        if vial_id not in vials:
            logger.error(f"Prompt error: Invalid vial {vial_id}")
            return jsonify({"detail": f"Invalid vial: {vial_id}"}), 400
        data = request.get_json()
        prompt = data.get('prompt')
        block_hash = data.get('blockHash')
        vials[vial_id]['tasks'].append({"prompt": prompt, "hash": block_hash})
        vials[vial_id]['status'] = 'running'
        logger.info(f"Prompt sent to {vial_id}: {prompt}")
        return jsonify({"response": f"Prompt processed for {vial_id}"})
    except Exception as e:
        logger.error(f"Prompt error for {vial_id}: {str(e)}")
        return jsonify({"detail": f"Prompt processing failed: {str(e)}"}), 500

@app.route('/api/vials/<vial_id>/task', methods=['POST'])
def send_task(vial_id):
    try:
        if vial_id not in vials:
            logger.error(f"Task error: Invalid vial {vial_id}")
            return jsonify({"detail": f"Invalid vial: {vial_id}"}), 400
        data = request.get_json()
        task = data.get('task')
        block_hash = data.get('blockHash')
        vials[vial_id]['tasks'].append(task)
        vials[vial_id]['status'] = 'running'
        logger.info(f"Task assigned to {vial_id}: {task}")
        return jsonify({"status": f"Task assigned to {vial_id}"})
    except Exception as e:
        logger.error(f"Task error for {vial_id}: {str(e)}")
        return jsonify({"detail": f"Task processing failed: {str(e)}"}), 500

@app.route('/api/vials/<vial_id>/config', methods=['PUT'])
def set_config(vial_id):
    try:
        if vial_id not in vials:
            logger.error(f"Config error: Invalid vial {vial_id}")
            return jsonify({"detail": f"Invalid vial: {vial_id}"}), 400
        data = request.get_json()
        key = data.get('key')
        value = data.get('value')
        block_hash = data.get('blockHash')
        vials[vial_id]['config'][key] = value
        vials[vial_id]['status'] = 'running'
        logger.info(f"Config set for {vial_id}: {key}={value}")
        return jsonify({"status": f"Config updated for {vial_id}"})
    except Exception as e:
        logger.error(f"Config error for {vial_id}: {str(e)}")
        return jsonify({"detail": f"Config update failed: {str(e)}"}), 500

@app.route('/api/vials/void', methods=['DELETE'])
def void_vials():
    try:
        for vial_id in vials:
            vials[vial_id] = {'status': 'stopped', 'tasks': [], 'config': {}}
        logger.info("All vials reset")
        return jsonify({"status": "All vials reset"})
    except Exception as e:
        logger.error(f"Void error: {str(e)}")
        return jsonify({"detail": f"Void failed: {str(e)}"}), 500

@app.route('/api/wallet/create', methods=['POST'])
def create_wallet():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        wallet = {
            'address': data.get('address', f"wallet-{uuid.uuid4()}"),
            'balance': data.get('balance', 0),
            'hash': data.get('hash', hashlib.sha256(str(datetime.utcnow()).encode()).hexdigest()),
            'webxos': data.get('webxos', 0.0000),
            'transactions': data.get('transactions', [])
        }
        wallets[user_id] = wallet
        logger.info(f"Wallet created for user {user_id}")
        return jsonify({"status": "Wallet created", "address": wallet['address']})
    except Exception as e:
        logger.error(f"Wallet creation error: {str(e)}")
        return jsonify({"detail": f"Wallet creation failed: {str(e)}"}), 500

@app.route('/api/wallet/import', methods=['POST'])
def import_wallet():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        wallet = {
            'address': data.get('address'),
            'balance': data.get('balance', 0),
            'hash': data.get('hash'),
            'webxos': data.get('webxos', 0.0000),
            'transactions': data.get('transactions', [])
        }
        wallets[user_id] = wallet
        logger.info(f"Wallet imported for user {user_id}")
        return jsonify({"status": "Wallet imported"})
    except Exception as e:
        logger.error(f"Wallet import error: {str(e)}")
        return jsonify({"detail": f"Wallet import failed: {str(e)}"}), 500

@app.route('/api/wallet/transaction', methods=['POST'])
def wallet_transaction():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        if user_id not in wallets:
            logger.error(f"Transaction error: No wallet for user {user_id}")
            return jsonify({"detail": "No wallet found"}), 400
        wallets[user_id]['transactions'].append({
            'type': data.get('type', 'transaction'),
            'timestamp': datetime.utcnow().isoformat()
        })
        wallets[user_id]['webxos'] += 0.0001
        logger.info(f"Transaction recorded for user {user_id}")
        return jsonify({"status": "Transaction recorded"})
    except Exception as e:
        logger.error(f"Wallet transaction error: {str(e)}")
        return jsonify({"detail": f"Wallet transaction failed: {str(e)}"}), 500

@app.route('/api/quantum/link', methods=['POST'])
def quantum_link():
    try:
        data = request.get_json()
        vial_ids = data.get('vials', [])
        statuses = ['running' if vid in vials else 'stopped' for vid in vial_ids]
        latencies = [50 + i * 10 for i in range(len(vial_ids))]
        logger.info("Quantum link established")
        return jsonify({"statuses": statuses, "latencies": latencies})
    except Exception as e:
        logger.error(f"Quantum link error: {str(e)}")
        return jsonify({"detail": f"Quantum link failed: {str(e)}"}), 500

@app.route('/api/blockchain/transaction', methods=['POST'])
def blockchain_transaction():
    try:
        data = request.get_json()
        blockchain.append(data)
        logger.info(f"Blockchain transaction recorded: {data['type']}")
        return jsonify({"status": "Transaction recorded"})
    except Exception as e:
        logger.error(f"Blockchain error: {str(e)}")
        return jsonify({"detail": f"Blockchain error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
