from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import uuid
import os

vials = [
    {"id": f"vial{i+1}", "status": "stopped", "code": "", "wallet": {"address": None, "balance": 0.0}, "tasks": []}
    for i in range(4)
]
wallet = {"address": None, "balance": 0.0}
network_id = None

class VialHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/mcp/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "offline"}).encode())

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length).decode())

        if self.path == '/mcp/auth':
            global network_id, wallet
            network_id = post_data.get('networkId', str(uuid.uuid4()))
            wallet['address'] = str(uuid.uuid4())
            for vial in vials:
                vial['wallet']['address'] = str(uuid.uuid4())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"token": "offline", "address": wallet['address']}).encode())

        elif self.path == '/mcp/train':
            code = post_data.get('code', '')
            is_python = post_data.get('isPython', True)
            for vial in vials:
                vial['code'] = code
                vial['status'] = 'running'
                vial['wallet']['balance'] += 0.0001
                vial['tasks'].append(f"task_{uuid.uuid4()}")
            wallet['balance'] += 0.0004
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"vials": vials, "balance": 0.0004}).encode())

        elif self.path == '/mcp/upload':
            file_path = f"/uploads/{post_data.get('filename', 'mock')}"
            with open(file_path, 'w') as f:
                f.write(post_data.get('code', ''))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"filePath": file_path}).encode())

        elif self.path == '/mcp/void':
            global network_id, wallet
            network_id = None
            wallet = {"address": None, "balance": 0.0}
            for vial in vials:
                vial.update({"status": "stopped", "code": "", "wallet": {"address": None, "balance": 0.0}, "tasks": []})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "voided"}).encode())

if __name__ == '__main__':
    os.makedirs('/uploads', exist_ok=True)
    server = HTTPServer(('0.0.0.0', 8080), VialHandler)
    server.serve_forever()
