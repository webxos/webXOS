from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import time
import uuid

app = Flask(__name__)
CORS(app, resources={r"/chatbot/*": {"origins": "*"}})

# Mock data
vial_states = {
    "vial1": {"status": "active", "pattern": "helix"},
    "vial2": {"status": "active", "pattern": "cube"},
    "vial3": {"status": "active", "pattern": "torus"},
    "vial4": {"status": "active", "pattern": "star"}
}
site_index = [
    {"path": "/app1", "source": "App1", "text": {"content": "Sample app", "keywords": ["app", "sample"]}},
    {"path": "/app2", "source": "App2", "text": {"content": "AI tool", "keywords": ["ai", "tool"]}}
]
error_log_file = "errorlog.md"

@app.route('/chatbot/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "timestamp": time.time()}), 200

@app.route('/chatbot/authenticate', methods=['POST'])
def authenticate():
    data = request.get_json()
    if not data or data.get('network') != 'webxos':
        return jsonify({"error": "Invalid network ID"}), 400
    token = str(uuid.uuid4())
    return jsonify({"token": token}), 200

@app.route('/chatbot/train_vials', methods=['POST'])
def train_vials():
    if not request.headers.get('Authorization'):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    if not data or not data.get('content') or not data.get('filename'):
        return jsonify({"error": "Invalid request"}), 400
    content = data['content']
    filename = data['filename']
    if not filename.endswith('.md') or '## Vial Data' not in content or 'wallet' not in content:
        return jsonify({"error": "Invalid .md format"}), 400
    json_block = content.split('```json\n')[1].split('\n```')[0] if '```json\n' in content else None
    if not json_block:
        return jsonify({"error": "Missing JSON block"}), 400
    try:
        json.loads(json_block)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in .md"}), 400
    return jsonify({"balance_earned": 100}), 200

@app.route('/chatbot/reset_vials', methods=['POST'])
def reset_vials():
    if not request.headers.get('Authorization'):
        return jsonify({"error": "Unauthorized"}), 401
    global vial_states
    vial_states = {
        "vial1": {"status": "active", "pattern": "helix"},
        "vial2": {"status": "active", "pattern": "cube"},
        "vial3": {"status": "active", "pattern": "torus"},
        "vial4": {"status": "active", "pattern": "star"}
    }
    return jsonify({"status": "vials reset"}), 200

@app.route('/chatbot/get_vials', methods=['GET'])
def get_vials():
    if not request.headers.get('Authorization'):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(vial_states), 200

@app.route('/chatbot/galaxy_search', methods=['POST'])
def galaxy_search():
    if not request.headers.get('Authorization'):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    query = data.get('query', '')
    vials = data.get('vials', [])
    if not query:
        return jsonify({"error": "Query required"}), 400
    results = [
        {"item": item, "matches": [{"value": item['text']['content'], "indices": [[0, len(query)]]}]}
        for item in site_index if query.lower() in item['text']['content'].lower()
    ]
    return jsonify(results), 200

@app.route('/chatbot/dna_reasoning', methods=['POST'])
def dna_reasoning():
    if not request.headers.get('Authorization'):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    query = data.get('query', '')
    vials = data.get('vials', [])
    if not query:
        return jsonify({"error": "Query required"}), 400
    results = [f"DNA reasoning result for '{query}' with vials {vials}"]
    return jsonify(results), 200

@app.route('/chatbot/log_error', methods=['POST'])
def log_error():
    data = request.get_json()
    if not data or not data.get('timestamp') or not data.get('message'):
        return jsonify({"error": "Invalid error log"}), 400
    with open(error_log_file, 'a') as f:
        f.write(f"[{data['timestamp']}] {data['message']}\n")
    return jsonify({"status": "error logged"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
