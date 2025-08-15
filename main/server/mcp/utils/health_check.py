from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": "2025-08-15T05:15:00Z",
            "services": {
                "auth": "running",
                "troubleshoot": "running"
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
