const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const http = require('http');

const app = express();
const server = http.createServer(app);
app.use(cors({ origin: 'https://webxos.netlify.app' }));
app.use(express.json());

const games = {};

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.post('/create', (req, res) => {
    const { gameName, port, maxPlayers } = req.body;
    if (!gameName || !port || !maxPlayers) {
        return res.status(400).json({ error: 'Invalid input' });
    }
    if (games[gameName]) {
        return res.status(400).json({ error: 'Game exists' });
    }
    const wss = new WebSocket.Server({ port });
    const players = {};
    wss.on('connection', ws => {
        ws.on('message', data => {
            const player = JSON.parse(data);
            players[player.id] = player;
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify(players));
                }
            });
        });
        ws.on('close', () => {
            delete players[player.id];
        });
    });
    games[gameName] = { port, maxPlayers, wss };
    res.json({ message: `Game ${gameName} created on port ${port}` });
});

server.listen(3000, () => console.log('Node.js on port 3000'));
