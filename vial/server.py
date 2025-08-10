from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import uuid
import os

app = Flask(__name__)

vials = [
    {"id": f"vial{i+1}", "status": "stopped", "code": "", "wallet": {"address": None, "balance": 0.0}, "tasks": []}
    for i in range(4)
]
wallet = {"address": None, "balance": 0.0}
network_id = None

@app.route('/mcp/ping', methods=['GET'])
def ping():
    return jsonify({"status": "online"}), 200

@app.route('/mcp/auth', methods=['POST'])
def auth():
    global network_id, wallet
    data = request.json
    network_id = data.get('networkId', str(uuid.uuid4()))
    wallet['address'] = str(uuid.uuid4())
    for vial in vials:
        vial['wallet']['address'] = str(uuid.uuid4())
    return jsonify({"token": str(uuid.uuid4()), "address": wallet['address']}), 200

@app.route('/mcp/train', methods=['POST'])
def train():
    if not network_id:
        return jsonify({"error": "Not authenticated"}), 401
    code = request.form.get('code')
    is_python = request.form.get('isPython') == 'true'
    for vial in vials:
        vial['code'] = code
        vial['status'] = 'running'
        vial['wallet']['balance'] += 0.0001
        vial['tasks'].append(f"task_{uuid.uuid4()}")
    wallet['balance'] += 0.0004
    return jsonify({"vials": vials, "balance": 0.0004}), 200

@app.route('/mcp/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file_path = os.path.join('/uploads', file.filename)
    file.save(file_path)
    return jsonify({"filePath": file_path}), 200

@app.route('/mcp/void', methods=['POST'])
def void():
    global network_id, wallet
    network_id = None
    wallet = {"address": None, "balance": 0.0}
    for vial in vials:
        vial.update({"status": "stopped", "code": "", "wallet": {"address": None, "balance": 0.0}, "tasks": []})
    return jsonify({"status": "voided"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
