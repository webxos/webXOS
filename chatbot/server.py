import http.server
import socketserver
import json
import torch
import pickle
from urllib.parse import parse_qs
from io import BytesIO

# Load model and vocab
VOCAB = pickle.load(open("chatbot/model/vocab.pkl", "rb"))
encoder = Encoder(len(VOCAB), 128)  # From previous model code
decoder = Decoder(128, len(VOCAB))
checkpoint = torch.load("chatbot/model/chatbot_model.pt")
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()

# Simple tokenization
def tokenize(text):
    return [VOCAB.get(word, VOCAB["<PAD>"]) for word in text.lower().split()]

# Chatbot inference
def generate_response(input_text):
    input_seq = torch.tensor([tokenize(input_text)], dtype=torch.long)
    with torch.no_grad():
        _, hidden = encoder(input_seq)
        output = []
        decoder_input = torch.tensor([[VOCAB["<SOS>"]]])
        for _ in range(20):  # Max response length
            decoder_output, hidden = decoder(decoder_input, hidden)
            _, topi = decoder_output.topk(1)
            token = topi.item()
            if token == VOCAB["<EOS>"]:
                break
            output.append(token)
            decoder_input = torch.tensor([[token]])
    return " ".join([word for word, idx in VOCAB.items() if idx in output])

# HTTP server
class ChatbotHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        message = data.get("message", "")

        # Generate response
        response = generate_response(message)

        # Send response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"response": response}).encode('utf-8'))

    def do_GET(self):
        # Serve chatbot.html
        if self.path == '/chatbot' or self.path == '/chatbot/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            with open("chatbot/chatbot.html", "rb") as f:
                self.wfile.write(f.read())
        # Serve static files
        elif self.path.startswith('/chatbot/static/'):
            try:
                file_path = self.path[1:]  # Remove leading slash
                self.send_response(200)
                if file_path.endswith('.css'):
                    self.send_header('Content-Type', 'text/css')
                elif file_path.endswith('.js'):
                    self.send_header('Content-Type', 'application/javascript')
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

# Run server
PORT = 8000
with socketserver.TCPServer(("", PORT), ChatbotHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()