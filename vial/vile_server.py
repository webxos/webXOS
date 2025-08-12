from flask import Flask, request, jsonify
from vial_manager import VialManager
from auth_manager import AuthManager
from webxos_wallet import WebXOSWallet
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, filename='vial.log')
logger = logging.getLogger(__name__)

wallet = WebXOSWallet()
vial_manager = VialManager(wallet)
auth_manager = AuthManager()

@app.route('/vial/authenticate', methods=['POST'])
def authenticate():
    try:
        data = request.get_json()
        if not data or 'network' not in data or 'session' not in data:
            logger.error('Invalid authentication request')
            return jsonify({'error': 'Invalid request'}), 400
        token, address = auth_manager.authenticate(data['network'], data['session'])
        logger.info(f"Authenticated {data['network']} with token {token}")
        return jsonify({'token': token, 'address': address})
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- [{new Date().toISOString()}] Authentication error: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/vial/train_vials', methods=['POST'])
def train_vials():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        if not auth_manager.validate_token(request.headers['Authorization'].replace('Bearer ', '')):
            logger.error('Invalid token')
            return jsonify({'error': 'Invalid token'}), 401
        data = request.get_json()
        balance_earned = vial_manager.train_vials(data['network_id'], data['content'], data['filename'])
        return jsonify({'balance_earned': balance_earned})
    except Exception as e:
        logger.error(f"Train vials error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- [{new Date().toISOString()}] Train vials error: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/vial/reset_vials', methods=['POST'])
def reset_vials():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        if not auth_manager.validate_token(request.headers['Authorization'].replace('Bearer ', '')):
            logger.error('Invalid token')
            return jsonify({'error': 'Invalid token'}), 401
        vial_manager.reset_vials()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Reset vials error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- [{new Date().toISOString()}] Reset vials error: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/vial/get_vials', methods=['GET'])
def get_vials():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        if not auth_manager.validate_token(request.headers['Authorization'].replace('Bearer ', '')):
            logger.error('Invalid token')
            return jsonify({'error': 'Invalid token'}), 401
        return jsonify(vial_manager.get_vials())
    except Exception as e:
        logger.error(f"Get vials error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- [{new Date().toISOString()}] Get vials error: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/vial/galaxy_search', methods=['POST'])
def galaxy_search():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        if not auth_manager.validate_token(request.headers['Authorization'].replace('Bearer ', '')):
            logger.error('Invalid token')
            return jsonify({'error': 'Invalid token'}), 401
        data = request.get_json()
        results = vial_manager.galaxy_search(data['query'], data['vials'])
        return jsonify(results)
    except Exception as e:
        logger.error(f"Galaxy search error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- [{new Date().toISOString()}] Galaxy search error: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/vial/dna_reasoning', methods=['POST'])
def dna_reasoning():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        if not auth_manager.validate_token(request.headers['Authorization'].replace('Bearer ', '')):
            logger.error('Invalid token')
            return jsonify({'error': 'Invalid token'}), 401
        data = request.get_json()
        results = vial_manager.dna_reasoning(data['query'], data['vials'])
        return jsonify(results)
    except Exception as e:
        logger.error(f"DNA reasoning error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- [{new Date().toISOString()}] DNA reasoning error: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/vial/log_error', methods=['POST'])
def log_error():
    try:
        data = request.get_json()
        with open("errorlog.md", "a") as f:
            f.write(f"- [{data['timestamp']}] {data['message']}\n")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Log error failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
