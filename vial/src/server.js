const express = require('express');
const WebSocket = require('ws');
const sqlite3 = require('sqlite3').verbose();
const jwt = require('jsonwebtoken');
const { OAuth2Client } = require('google-auth-library');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const port = process.env.PORT || 8080;
const dbPath = '/vial/database.db';
const errorLogPath = '/vial/errorlog.md';
const secret = process.env.OAUTH_CLIENT_SECRET || 'default_secret';
let db;

async function logError(message, analysis, stack, urgency) {
    const timestamp = new Date().toISOString();
    const errorMessage = `[${timestamp}] ERROR: ${message}\nAnalysis: ${analysis}\nTraceback: ${stack || 'No stack'}\n---\n`;
    try {
        await fs.appendFile(errorLogPath, errorMessage);
    } catch (err) {
        console.error(`Failed to write to errorlog.md: ${err.message}`);
    }
}

async function initDb() {
    try {
        db = new sqlite3.Database(dbPath, (err) => {
            if (err) throw new Error(`DB Init Error: ${err.message}`);
        });
        await db.run('CREATE TABLE IF NOT EXISTS vials (id TEXT PRIMARY KEY, status TEXT, code TEXT, filePath TEXT, createdAt TEXT, codeLength INTEGER, latencyHistory TEXT)');
        await db.run('CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, event_type TEXT, message TEXT, metadata TEXT, urgency TEXT)');
        await logError('Database Initialized', 'SQLite database created at /vial/database.db', 'No stack', 'INFO');
    } catch (err) {
        await logError(`DB Init Error: ${err.message}`, 'Check /vial/src/server.js:20 or file permissions', err.stack || 'No stack', 'CRITICAL');
        throw err;
    }
}

app.use(express.json());
app.use('/vial/static', express.static('/vial/static'));

app.get('/mcp/health', async (req, res) => {
    try {
        res.status(200).json({ status: 'healthy' });
    } catch (err) {
        await logError(`Health Check Error: ${err.message}`, 'Check /vial/src/server.js:30', err.stack || 'No stack', 'HIGH');
        res.status(500).json({ error: 'Server error' });
    }
});

app.post('/mcp/auth', async (req, res) => {
    try {
        const { token } = req.body;
        const client = new OAuth2Client();
        const ticket = await client.verifyIdToken({ idToken: token, audience: secret });
        const payload = ticket.getPayload();
        const newToken = jwt.sign({ sub: payload.sub, exp: Math.floor(Date.now() / 1000) + 3600 }, secret);
        res.json({ token: newToken });
    } catch (err) {
        await logError(`Auth Error: ${err.message}`, 'Check /vial/src/oauth.js:20 or /vial/mcp.json:10', err.stack || 'No stack', 'HIGH');
        res.status(401).json({ error: 'Authentication failed' });
    }
});

app.post('/mcp/vial', async (req, res) => {
    try {
        const { id, code, training, agentId } = req.body;
        const token = req.headers.authorization?.split(' ')[1];
        if (!token || !jwt.verify(token, secret)) throw new Error('Invalid token');
        const createdAt = new Date().toISOString();
        const codeLength = code.js.length;
        await db.run('INSERT INTO vials (id, status, code, filePath, createdAt, codeLength, latencyHistory) VALUES (?, ?, ?, ?, ?, ?, ?)',
            [id, 'running', JSON.stringify(code), `/vial/uploads/vial${id}.js`, createdAt, codeLength, JSON.stringify([50])]);
        await logError(`Vial Created: ${id}`, 'Vial initialized in /vial/src/vial_manager.js:40', 'No stack', 'INFO');
        res.json({ id, latency: 50, createdAt, codeLength });
    } catch (err) {
        await logError(`Vial Creation Error: ${err.message}`, 'Check /vial/src/vial_manager.js:40 or /vial/src/server.js:50', err.stack || 'No stack', 'HIGH');
        res.status(500).json({ error: 'Vial creation failed' });
    }
});

app.get('/mcp/vials', async (req, res) => {
    try {
        const token = req.headers.authorization?.split(' ')[1];
        if (!token || !jwt.verify(token, secret)) throw new Error('Invalid token');
        db.all('SELECT * FROM vials', [], (err, rows) => {
            if (err) throw err;
            res.json(rows.map(row => ({
                id: row.id, status: row.status, code: JSON.parse(row.code), filePath: row.filePath, createdAt: row.createdAt, codeLength: row.codeLength, latencyHistory: JSON.parse(row.latencyHistory)
            })));
        });
    } catch (err) {
        await logError(`Vials Fetch Error: ${err.message}`, 'Check /vial/src/vial_manager.js:50 or /vial/src/server.js:60', err.stack || 'No stack', 'HIGH');
        res.status(500).json({ error: 'Failed to fetch vials' });
    }
});

app.post('/mcp/train', async (req, res) => {
    try {
        const { id, input, agentId } = req.body;
        const token = req.headers.authorization?.split(' ')[1];
        if (!token || !jwt.verify(token, secret)) throw new Error('Invalid token');
        db.get('SELECT * FROM vials WHERE id = ?', [id], async (err, row) => {
            if (err || !row) throw new Error('Vial not found');
            const latencyHistory = JSON.parse(row.latencyHistory);
            latencyHistory.push(50 + Math.random() * 10);
            await db.run('UPDATE vials SET latencyHistory = ? WHERE id = ?', [JSON.stringify(latencyHistory), id]);
            await logError(`Vial Trained: ${id}`, 'Training completed in /vial/src/training.js:20', 'No stack', 'INFO');
            res.json({ latency: latencyHistory[latencyHistory.length - 1], codeLength: input.length });
        });
    } catch (err) {
        await logError(`Train Error: ${err.message}`, 'Check /vial/src/training.js:20 or /vial/src/server.js:70', err.stack || 'No stack', 'HIGH');
        res.status(500).json({ error: 'Training failed' });
    }
});

app.post('/mcp/destroy', async (req, res) => {
    try {
        const token = req.headers.authorization?.split(' ')[1];
        if (!token || !jwt.verify(token, secret)) throw new Error('Invalid token');
        await db.run('DELETE FROM vials');
        await logError('All Vials Destroyed', 'Vials cleared in /vial/src/vial_manager.js:60', 'No stack', 'INFO');
        res.json({ status: 'success' });
    } catch (err) {
        await logError(`Destroy Error: ${err.message}`, 'Check /vial/src/vial_manager.js:60 or /vial/src/server.js:80', err.stack || 'No stack', 'HIGH');
        res.status(500).json({ error: 'Failed to destroy vials' });
    }
});

app.post('/mcp/log-sync', async (req, res) => {
    try {
        const { log } = req.body;
        const token = req.headers.authorization?.split(' ')[1];
        if (!token || !jwt.verify(token, secret)) throw new Error('Invalid token');
        const logData = JSON.parse(log);
        await db.run('INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
            [logData.timestamp, logData.event_type, logData.message, JSON.stringify(logData.metadata), logData.urgency]);
        await logError('Log Synced', 'Log saved in /vial/src/log_manager.js:50', 'No stack', 'INFO');
        res.json({ status: 'success' });
    } catch (err) {
        await logError(`Log Sync Error: ${err.message}`, 'Check /vial/src/log_manager.js:50 or /vial/src/server.js:90', err.stack || 'No stack', 'HIGH');
        res.status(500).json({ error: 'Log sync failed' });
    }
});

app.get('/mcp/diagnostics', async (req, res) => {
    try {
        const token = req.headers.authorization?.split(' ')[1];
        if (!token || !jwt.verify(token, secret)) throw new Error('Invalid token');
        const issues = [];
        try { await fs.access(dbPath); } catch { issues.push({ message: 'Database file missing', analysis: 'Check /vial/database.db', stack: 'No stack' }); }
        try { await fs.access(errorLogPath); } catch { issues.push({ message: 'Error log file missing', analysis: 'Check /vial/errorlog.md', stack: 'No stack' }); }
        res.json({ issues });
    } catch (err) {
        await logError(`Diagnostics Error: ${err.message}`, 'Check /vial/src/diagnostics.js:40 or /vial/src/server.js:100', err.stack || 'No stack', 'HIGH');
        res.status(500).json({ error: 'Diagnostics failed' });
    }
});

const server = app.listen(port, async () => {
    try {
        await initDb();
        await logError(`Server Started on port ${port}`, 'Server initialized in /vial/src/server.js:110', 'No stack', 'INFO');
    } catch (err) {
        await logError(`Server Start Error: ${err.message}`, 'Check /vial/src/server.js:110 or port availability', err.stack || 'No stack', 'CRITICAL');
        process.exit(1);
    }
});

const wss = new WebSocket.Server({ server });
wss.on('connection', (ws, req) => {
    try {
        const token = new URLSearchParams(req.url.split('?')[1]).get('token');
        if (!token || !jwt.verify(token, secret)) throw new Error('Invalid token');
        ws.on('message', async (message) => {
            try {
                const data = JSON.parse(message);
                await logError(`WebSocket Message: ${data.message}`, 'Processed in /vial/src/server.js:120', 'No stack', 'INFO');
                wss.clients.forEach(client => client.send(JSON.stringify(data)));
            } catch (err) {
                await logError(`WebSocket Message Error: ${err.message}`, 'Check /vial/src/server.js:120', err.stack || 'No stack', 'HIGH');
            }
        });
        ws.on('close', () => logError('WebSocket Disconnected', 'Client disconnected in /vial/src/server.js:130', 'No stack', 'INFO'));
    } catch (err) {
        await logError(`WebSocket Error: ${err.message}`, 'Check /vial/src/server.js:130 or OAuth', err.stack || 'No stack', 'HIGH');
        ws.close();
    }
});

process.on('uncaughtException', async (err) => {
    await logError(`Uncaught Exception: ${err.message}`, 'Check /vial/src/server.js:140', err.stack || 'No stack', 'CRITICAL');
    process.exit(1);
});
