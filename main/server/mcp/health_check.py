from flask import Flask, jsonify
import psutil
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        if cpu_usage > 90 or memory.percent > 90:
            return jsonify({
                "status": "unhealthy",
                "details": {
                    "cpu_usage": f"{cpu_usage}%",
                    "memory_usage": f"{memory.percent}%"
                }
            }), 503
        return jsonify({
            "status": "healthy",
            "timestamp": "2025-08-15T05:45:00Z",
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
    app.run(host='0.0.0.0', port=8081)from flask import Flask, jsonify
import psutil
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        if cpu_usage > 90 or memory.percent > 90:
            return jsonify({
                "status": "unhealthy",
                "details": {
                    "cpu_usage": f"{cpu_usage}%",
                    "memory_usage": f"{memory.percent}%"
                }
            }), 503
        return jsonify({
            "status": "healthy",
            "timestamp": "2025-08-15T05:45:00Z",
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
