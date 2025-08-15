from flask import Flask, request, jsonify
from server.mcp.utils.error_handler import ErrorHandler
import json

app = Flask(__name__)

def validate_response(response):
    if not response or not isinstance(response, (dict, list, str)):
        return ErrorHandler.handle_error("Invalid response format", "Response validation failed")
    try:
        if isinstance(response, str):
            return json.loads(response)
        return response
    except json.JSONDecodeError as e:
        return ErrorHandler.handle_error(e, "JSON decode error")

@app.route('/vial2/api/troubleshoot', methods=['POST'])
def troubleshoot():
    try:
        # Simulate troubleshoot response
        response = {"status": "OK", "details": "System check completed"}
        validated_response = validate_response(response)
        return jsonify(validated_response), 200
    except Exception as e:
        return jsonify(ErrorHandler.handle_error(e)), 500

@app.route('/vial2/api/auth/oauth', methods=['POST'])
def oauth():
    try:
        data = request.get_json()
        if not data or not data.get('provider') or not data.get('code'):
            return jsonify(ErrorHandler.handle_error("Missing required fields", "OAuth request validation")), 400
        # Simulate OAuth response
        if data.get('provider') == 'mock' and data.get('code') == 'test_code':
            response = {"access_token": "mock_token_xyz", "vials": ["vial1"]}
            return jsonify(validate_response(response)), 200
        return jsonify(ErrorHandler.handle_error("Invalid credentials")), 401
    except Exception as e:
        return jsonify(ErrorHandler.handle_error(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
