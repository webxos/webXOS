from flask import Flask, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.errorhandler(404)
def not_found(error):
    logging.error(f"404 Error: {request.path} at {datetime.datetime.utcnow().isoformat()}")
    return jsonify({
        "error": {
            "code": -32004,
            "message": f"Endpoint {request.path} not found",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    }), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
