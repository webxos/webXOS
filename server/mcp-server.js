```javascript
     const express = require('express');
     const http = require('http');
     const WebSocket = require('ws');

     const app = express();
     const server = http.createServer(app);
     const wss = new WebSocket.Server({ server });

     app.use(express.static('static'));

     wss.on('connection', (ws) => {
         console.log('Client connected');
         ws.send(JSON.stringify({ event: 'server-started', data: { message: 'MCP server started' } }));

         ws.on('message', (message) => {
             const msg = JSON.parse(message.toString());
             console.log('Received:', msg);
             if (msg.event === 'start-server') {
                 ws.send(JSON.stringify({ event: 'server-started', data: { message: 'MCP server started' } }));
             } else if (msg.event === 'end-server') {
                 ws.send(JSON.stringify({ event: 'server-stopped', data: { message: 'MCP server stopped' } }));
             } else if (msg.event === 'check-agents') {
                 const agents = [
                     { agent: 'agent1', status: 'online', latency: Math.random() * 100 },
                     { agent: 'agent2', status: 'online', latency: Math.random() * 100 },
                     { agent: 'agent3', status: 'offline', latency: 0 },
                     { agent: 'agent4', status: 'online', latency: Math.random() * 100 }
                 ];
                 ws.send(JSON.stringify({ event: 'agent-status', data: agents }));
             }
         });

         ws.on('close', () => {
             console.log('Client disconnected');
         });

         ws.on('error', (err) => {
             console.error('WebSocket error:', err);
         });
     });

     server.listen(8080, () => {
         console.log('Server running at http://localhost:8080');
     });
     ```
