/**
 * Node.js server for Vial MCP Controller
 * Dependencies: express, sqlite3, ws, jsonwebtoken
 * Handles API endpoints, WebSocket, and SQLite database
 * Rebuild: Install dependencies with `npm install express sqlite3 ws jsonwebtoken`
 */
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const WebSocket = require('ws');
const jwt = require('jsonwebtoken');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.PORT || 8080;
const dbPath = process.env.DB_PATH || './vial.db';
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '../mcp.json')));
const secret = process.env.OAUTH_CLIENT_SECRET || 'your_client_secret';

app.use(express.json());
app.use('/vial/static', express.static(path.join(__dirname, '../static')));

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) console.error(`[ERROR] Database Init: ${err.message}`);
  db.run(`
    CREATE TABLE IF NOT EXISTS vials (
      id TEXT PRIMARY KEY,
      code TEXT,
      status TEXT,
      latencyHistory TEXT,
      filePath TEXT,
      createdAt TEXT,
      codeLength INTEGER
    );
    CREATE TABLE IF NOT EXISTS logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT,
      event_type TEXT,
      message TEXT,
      metadata TEXT,
      urgency TEXT
    );
  `);
});

const wss = new WebSocket.Server({ port: 8080, path: '/ws' });

wss.on('connection', (ws, req) => {
  try {
    const token = req.url.split('token=')[1];
    if (!token || !jwt.verify(token, secret)) {
      ws.close();
      return;
    }
    ws.on('message', (data) => {
      try {
        const { event_type, message, metadata } = JSON.parse(data);
        db.run(
          'INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
          [new Date().toISOString(), event_type, message, JSON.stringify(metadata), 'INFO'],
          (err) => { if (err) console.error(`[ERROR] Log Save: ${err.message}`); }
        );
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) client.send(data);
        });
      } catch (err) {
        console.error(`[ERROR] WebSocket Message: ${err.message}`);
      }
    });
  } catch (err) {
    console.error(`[ERROR] WebSocket Connection: ${err.message}`);
    ws.close();
  }
});

app.post('/mcp/auth', (req, res) => {
  try {
    const { token } = req.body;
    if (token === 'anonymous') {
      const anonToken = jwt.sign({ scopes: ['vial:read'] }, secret, { expiresIn: '1h' });
      return res.json({ token: anonToken });
    }
    const decoded = jwt.verify(token, secret);
    if (decoded.scopes.includes('vial:write')) {
      return res.json({ token });
    }
    res.status(401).json({ error: 'Invalid token' });
  } catch (err) {
    res.status(500).json({ error: `Auth Error: ${err.message}` });
  }
});

app.get('/mcp/health', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    res.status(200).json({ status: 'ok' });
  } catch (err) {
    res.status(401).json({ error: `Health Check Error: ${err.message}` });
  }
});

app.post('/mcp/vial', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    const { id, code, training } = req.body;
    const vial = {
      id,
      code: JSON.stringify(code),
      status: 'running',
      latencyHistory: JSON.stringify([Math.random() * 100]),
      filePath: `/vial/uploads/vial${id}.js`,
      createdAt: new Date().toISOString(),
      codeLength: code.js.length
    };
    db.run(
      'INSERT INTO vials (id, code, status, latencyHistory, filePath, createdAt, codeLength) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [vial.id, vial.code, vial.status, vial.latencyHistory, vial.filePath, vial.createdAt, vial.codeLength],
      (err) => {
        if (err) throw new Error(`Vial Save Error: ${err.message}`);
        res.json(vial);
      }
    );
  } catch (err) {
    res.status(500).json({ error: `Vial Creation Error: ${err.message}` });
  }
});

app.get('/mcp/vials', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    db.all('SELECT * FROM vials', (err, rows) => {
      if (err) throw new Error(`Vial Fetch Error: ${err.message}`);
      res.json(rows.map(row => ({
        id: row.id,
        code: JSON.parse(row.code),
        status: row.status,
        latencyHistory: JSON.parse(row.latencyHistory),
        filePath: row.filePath,
        createdAt: row.createdAt,
        codeLength: row.codeLength
      })));
    });
  } catch (err) {
    res.status(500).json({ error: `Vial Fetch Error: ${err.message}` });
  }
});

app.post('/mcp/train', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    const { id, input } = req.body;
    db.get('SELECT * FROM vials WHERE id = ?', [id], (err, row) => {
      if (err || !row) throw new Error(`Vial Not Found: ${err?.message || 'Invalid ID'}`);
      const latency = Math.random() * 100;
      const codeLength = input.length;
      db.run(
        'UPDATE vials SET latencyHistory = ?, code = ?, codeLength = ? WHERE id = ?',
        [JSON.stringify([...JSON.parse(row.latencyHistory), latency]), JSON.stringify({ js: input }), codeLength, id],
        (err) => {
          if (err) throw new Error(`Vial Update Error: ${err.message}`);
          res.json({ latency, codeLength });
        }
      );
    });
  } catch (err) {
    res.status(500).json({ error: `Train Error: ${err.message}` });
  }
});

app.post('/mcp/destroy', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    db.run('DELETE FROM vials', (err) => {
      if (err) throw new Error(`Vial Destroy Error: ${err.message}`);
      res.json({ status: 'ok' });
    });
  } catch (err) {
    res.status(500).json({ error: `Destroy Error: ${err.message}` });
  }
});

app.get('/mcp/diagnostics', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    const issues = [];
    if (!fs.existsSync(dbPath)) issues.push({ message: 'Database file missing', analysis: 'Check DB_PATH in .env' });
    res.json({ issues });
  } catch (err) {
    res.status(500).json({ error: `Diagnostics Error: ${err.message}` });
  }
});

app.post('/mcp/log-sync', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    const { log } = req.body;
    db.run(
      'INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
      Object.values(JSON.parse(require('lz-string').decompressFromUTF16(log))),
      (err) => {
        if (err) throw new Error(`Log Sync Error: ${err.message}`);
        res.json({ status: 'ok' });
      }
    );
  } catch (err) {
    res.status(500).json({ error: `Log Sync Error: ${err.message}` });
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

// Rebuild Instructions: Place in /vial/src/. Install dependencies: `npm install express sqlite3 ws jsonwebtoken`. Ensure /vial/mcp.json and /vial/.env exist. Run with `node server.js`. Check /vial/errorlog.md for issues.
