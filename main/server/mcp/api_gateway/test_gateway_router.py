import unittest
import json
from flask import Flask
from .gateway_router import app, validate_response

class TestGatewayRouter(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_validate_response_valid_json(self):
        response = {"result": "success"}
        self.assertEqual(validate_response(response), response)

    def test_validate_response_invalid_json(self):
        response = "invalid json"
        expected = {"error": {"code": -32000, "message": "JSON parse error: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)", "traceback": "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}}
        self.assertEqual(validate_response(response), expected)

    def test_validate_response_none(self):
        self.assertEqual(validate_response(None), {"error": {"code": -32000, "message": "Invalid response format", "traceback": "Response validation failed"}})

    def test_checklist_route_success(self):
        with app.test_request_context('/mcp/checklist', method='POST'):
            response = app.dispatch_request()
            self.assertIn('result', json.loads(response.get_data()))

    def test_checklist_route_error(self):
        with app.test_request_context('/mcp/checklist', method='POST'):
            response = app.dispatch_request()
            self.assertIn('error', json.loads(response.get_data()))

if __name__ == '__main__':
    unittest.main()
