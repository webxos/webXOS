from flask import Flask, request, jsonify
import requests
import logging
from vial_manager import VialManager
from webxos_wallet import WebXOSWallet

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, filename='chatbot.log')
logger = logging.getLogger(__name__)

# Initialize VialManager
wallet = WebXOSWallet()
vial_manager = VialManager(wallet)

# Proxy to /vial/ backend
VIAL_BACKEND = 'http://localhost:5000/vial'  # Adjust to actual /vial/ server URL

@app.route('/chatbot/authenticate', methods=['POST'])
def authenticate():
    try:
        data = request.get_json()
        if not data or 'network' not in data or 'session' not in data:
            logger.error('Invalid authentication request')
            return jsonify({'error': 'Invalid request'}), 400
        response = requests.post(f'{VIAL_BACKEND}/authenticate', json=data)
        if response.status_code != 200:
            logger.error(f'Authentication failed: {response.text}')
            return jsonify({'error': 'Authentication failed'}), response.status_code
        return jsonify(response.json())
    except Exception as e:
        logger.error(f'Authentication error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/train_vials', methods=['POST'])
def train_vials():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        data = request.get_json()
        response = requests.post(f'{VIAL_BACKEND}/train_vials', json=data, headers={'Authorization': request.headers['Authorization']})
        if response.status_code != 200:
            logger.error(f'Train vials failed: {response.text}')
            return jsonify({'error': 'Train vials failed'}), response.status_code
        return jsonify(response.json())
    except Exception as e:
        logger.error(f'Train vials error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/reset_vials', methods=['POST'])
def reset_vials():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        response = requests.post(f'{VIAL_BACKEND}/reset_vials', headers={'Authorization': request.headers['Authorization']})
        if response.status_code != 200:
            logger.error(f'Reset vials failed: {response.text}')
            return jsonify({'error': 'Reset vials failed'}), response.status_code
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f'Reset vials error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/get_vials', methods=['GET'])
def get_vials():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        response = requests.get(f'{VIAL_BACKEND}/get_vials', headers={'Authorization': request.headers['Authorization']})
        if response.status_code != 200:
            logger.error(f'Get vials failed: {response.text}')
            return jsonify({'error': 'Get vials failed'}), response.status_code
        return jsonify(response.json())
    except Exception as e:
        logger.error(f'Get vials error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/galaxy_search', methods=['POST'])
def galaxy_search():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        data = request.get_json()
        # Mock galaxy search (replace with actual web crawl logic)
        results = [{'item': {'path': '/mock', 'source': 'mock', 'text': {'content': data['query'], 'keywords': [data['query']]}}, 'matches': [{'value': data['query'], 'indices': [[0, len(data['query'])]]}]}]
        return jsonify(results)
    except Exception as e:
        logger.error(f'Galaxy search error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/dna_reasoning', methods=['POST'])
def dna_reasoning():
    try:
        if not request.headers.get('Authorization'):
            logger.error('Missing Authorization header')
            return jsonify({'error': 'Unauthorized'}), 401
        data = request.get_json()
        # Mock DNA reasoning (replace with actual logic)
        results = ['Reasoned response for ' + data['query']]
        return jsonify(results)
    except Exception as e:
        logger.error(f'DNA reasoning error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/log_error', methods=['POST'])
def log_error():
    try:
        data = request.get_json()
        with open('errorlog.md', 'a') as f:
            f.write(f"- [{data['timestamp']}] {data['message']}\n")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f'Log error failed: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
