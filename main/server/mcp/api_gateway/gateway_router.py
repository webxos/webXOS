from flask import Flask, request, jsonify
from server.mcp.config.oauth_config import oauth_config
from server.mcp.utils.logging_config import logger
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
        logger.error(f"JSON Decode Error: {str(e)}")
        return ErrorHandler.handle_error(e, "JSON decode error")

@app.route('/vial2/api/troubleshoot', methods=['POST'])
def troubleshoot():
    try:
        response = {"status": "OK", "details": "System check completed at 06:15 AM EDT"}
        validated_response = validate_response(response)
        return jsonify(validated_response), 200
    except Exception as e:
        logger.error(f"Troubleshoot Error: {str(e)}")
        return jsonify(ErrorHandler.handle_error(e)), 500

@app.route('/vial2/api/auth/oauth', methods=['POST'])
def oauth():
    try:
        data = request.get_json()
        if not data or not data.get('provider') or not data.get('code'):
            logger.error("Missing required OAuth fields")
            return jsonify(ErrorHandler.handle_error("Missing required fields", "OAuth request validation")), 400
        if not oauth_config.validate_credentials(data['provider'], data['code']):
            logger.error(f"Invalid credentials for provider: {data['provider']}")
            return jsonify(ErrorHandler.handle_error("Invalid OAuth credentials")), 401
        response = {"access_token": "mock_token_jkl", "vials": ["vial1"]}
        return jsonify(validate_response(response)), 200
    except Exception as e:
        logger.error(f"OAuth Error: {str(e)}")
        return jsonify(ErrorHandler.handle_error(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
