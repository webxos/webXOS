```python
     from flask import Flask, send_from_directory
     from flask_socketio import SocketIO, emit
     import torch
     import random

     app = Flask(__name__)
     socketio = SocketIO(app, cors_allowed_origins="*")

     @app.route('/<path:path>')
     def static_files(path):
         return send_from_directory('static', path)

     @socketio.on('connect')
     def handle_connect():
         print('Client connected')
         emit('server-started', {'message': 'MCP server started'})

     @socketio.on('start-server')
     def handle_start_server():
         emit('server-started', {'message': 'MCP server started'})
         # Example PyTorch integration
         tensor = torch.tensor([1.0])
         prediction = torch.nn.Linear(1, 1)(tensor).item()
         print(f'PyTorch test prediction: {prediction}')

     @socketio.on('end-server')
     def handle_end_server():
         emit('server-stopped', {'message': 'MCP server stopped'})

     @socketio.on('check-agents')
     def handle_check_agents():
         agents = [
             {'agent': 'agent1', 'status': 'online', 'latency': random.random() * 100},
             {'agent': 'agent2', 'status': 'online', 'latency': random.random() * 100},
             {'agent': 'agent3', 'status': 'offline', 'latency': 0},
             {'agent': 'agent4', 'status': 'online', 'latency': random.random() * 100}
         ]
         emit('agent-status', agents)

     @socketio.on('disconnect')
     def handle_disconnect():
         print('Client disconnected')

     if __name__ == '__main__':
         socketio.run(app, host='0.0.0.0', port=8080)
     ```
