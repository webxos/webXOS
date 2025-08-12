from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/vial/*": {"origins": "*"}})

# Mock vial agent states
vial_agents = {
    "vial1": {"status": "active", "pattern": "helix"},
    "vial2": {"status": "active", "pattern": "cube"},
    "vial3": {"status": "active", "pattern": "torus"},
    "vial4": {"status": "active", "pattern": "star"}
}

@app.route('/vial/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "timestamp": time.time()}), 200

@app.route('/vial/get_agents', methods=['GET'])
def get_agents():
    if not request.headers.get('Authorization'):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(vial_agents), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
