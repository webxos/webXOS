from flask import Flask, jsonify, request
from flask_cors import CORS
import yaml
from qutip_sim import run_qutip_sim

app = Flask(__name__)
CORS(app)

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {'api': {'key': 'default-key'}, 'quantum': {'qubits': 20}, 'qutip': {'noise_level': 0.01}}

@app.route('/api/simulator', methods=['GET', 'POST'])
def quantum_simulator():
    if request.method == 'POST':
        try:
            data = request.get_json()
            circuit = data.get('circuit', 'bell')
            result = run_qutip_sim(circuit, config['quantum']['qubits'])
            return jsonify({'status': 'success', 'result': result})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'ready', 'qubits': config['quantum']['qubits']})

@app.route('/api/settings', methods=['GET'])
def settings():
    return jsonify({'status': 'success', 'api_key': config['api']['key']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)