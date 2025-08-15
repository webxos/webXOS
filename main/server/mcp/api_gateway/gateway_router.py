from flask import Flask, request, jsonify
from .service_registry import ServiceRegistry
import json

app = Flask(__name__)
registry = ServiceRegistry()

def validate_response(response):
    if not response or not isinstance(response, (dict, list, str)):
        return {"error": {"code": -32000, "message": "Invalid response format", "traceback": "Response validation failed"}}
    try:
        if isinstance(response, str):
            return json.loads(response)
        return response
    except json.JSONDecodeError as e:
        return {"error": {"code": -32000, "message": f"JSON parse error: {str(e)}", "traceback": str(e)}}

@app.route('/mcp/checklist', methods=['POST'])
def checklist():
    service = registry.get_service('checklist')
    if not service:
        return jsonify({"error": {"code": -32601, "message": "Service not found", "traceback": "No checklist service registered"}}), 404
    response = service()
    validated_response = validate_response(response)
    return jsonify(validated_response)

@app.route('/mcp/auth/oauth', methods=['POST'])
def oauth():
    service = registry.get_service('oauth')
    if not service:
        return jsonify({"error": {"code": -32601, "message": "Service not found", "traceback": "No oauth service registered"}}), 404
    data = request.get_json()
    response = service(data.get('provider'), data.get('code'))
    validated_response = validate_response(response)
    return jsonify(validated_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
