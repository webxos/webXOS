// src/server.js
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const Ajv = require('ajv');
const LZString = require('lz-string');
const vialManager = require('./tools/vial_manager');
const logManager = require('./tools/log_manager');
const apiConnector = require('./tools/api_connector');
const training = require('./tools/training');

const app = express();
const ajv = new Ajv();
app.use(express.json());
app.use('/static', express.static(path.join(__dirname, '../static')));
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// SQLite database
const db = new sqlite3.Database('./uploads/vial.db', (err) => {
    if (err) console.error(`[SERVER] SQLite Error: ${err.message}`);
    else {
        db.run(`
            CREATE TABLE IF NOT EXISTS vials (
                id TEXT PRIMARY KEY,
                code TEXT,
                training TEXT,
                status TEXT,
                latencyHistory TEXT,
                filePath TEXT,
                createdAt TEXT,
                codeLength INTEGER
            );
        `);
        db.run(`
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                message TEXT,
                metadata TEXT,
                urgency TEXT
            );
        `);
        console.log('[SERVER] SQLite connected');
    }
});

// Routes
app.get('/', (req, res) => res.sendFile(path.join(__dirname, '../static/vial.html')));
app.get('/mcp/tools', (req, res) => res.json({ tools: ['vial_manager', 'log_manager', 'api_connector', 'training'] }));
app.get('/mcp/vials', (req, res) => vialManager.getVials(db, req, res));
app.post('/mcp/vial', (req, res) => vialManager.createVial(db, ajv, req, res));
app.post('/mcp/train', (req, res) => training.trainVial(db, req, res));
app.post('/mcp/api', (req, res) => apiConnector.fetchData(req, res));
app.post('/mcp/destroy', (req, res) => vialManager.destroyAllVials(db, req, res));
app.post('/mcp/log-sync', (req, res) => logManager.syncLog(db, LZString, req, res));

app.listen(8080, () => console.log('[SERVER] Running on port 8080'));

// Instructions:
// - Single-file Node.js server with SQLite
// - Serves vial.html and MCP endpoints
// - Install: `npm install express sqlite3 ajv lz-string redaxios`
// - Database: ./uploads/vial.db
