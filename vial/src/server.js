/**
 * Node.js server for Vial MCP Controller with Sanitized APIs
 * Dependencies: express, sqlite3, ws, jsonwebtoken, dns, express-validator, express-rate-limit, axios
 * Handles sanitized REST APIs, WebSocket, SQLite, and AI model connectivity
 * Rebuild: Install dependencies with `npm install express sqlite3 ws jsonwebtoken dns express-validator express-rate-limit axios`
 */
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const WebSocket = require('ws');
const jwt = require('jsonwebtoken');
const fs = require('fs');
const path = require('path');
const dns = require('dns').promises;
const { body, validationResult } = require('express-validator');
const rateLimit = require('express-rate-limit');
const axios = require('axios');

const app = express();
const port = process.env.PORT || 8080;
const dbPath = process.env.DB_PATH || './vial.db';
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '../mcp.json')));
const secret = process.env.OAUTH_CLIENT_SECRET || 'your_client_secret';

// In-memory service registry (replace with Consul/etcd for production)
const serviceRegistry = new Map();

app.use(express.json());
app.use('/vial/static', express.static(path.join(__dirname, '../static')));

// Rate limiting for all endpoints
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit to 100 requests per window
  message: { error: 'Too many requests, please try again later.' }
});
app.use(limiter);

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error(`[ERROR] Database Init: ${err.message}`);
    logSecurityEvent('Database', 'Init Failure', err.message);
  }
  db.run(`
    CREATE TABLE IF NOT EXISTS vials (
      id TEXT PRIMARY KEY,
      code TEXT,
      status TEXT,
      latencyHistory TEXT,
      filePath TEXT,
      createdAt TEXT,
      codeLength INTEGER,
      agentId TEXT
    );
    CREATE TABLE IF NOT EXISTS logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT,
      event_type TEXT,
      message TEXT,
      metadata TEXT,
      urgency TEXT
    );
    CREATE TABLE IF NOT EXISTS agents (
      id TEXT PRIMARY KEY,
      name TEXT,
      endpoints TEXT,
      capabilities TEXT,
      status TEXT,
      lastPing TEXT
    );
  `);
});

const wss = new WebSocket.Server({ port: 8080, path: '/ws' });

wss.on('connection', (ws, req) => {
  try {
    const token = req.url.split('token=')[1];
    if (!token || !jwt.verify(token, secret)) {
      ws.close();
      logSecurityEvent('WebSocket', 'Unauthorized Connection', 'Invalid or missing token');
      return;
    }
    ws.on('message', (data) => {
      try {
        const { event_type, message, metadata } = JSON.parse(data);
        db.run(
          'INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
          [new Date().toISOString(), event_type, message, JSON.stringify(sanitizeMetadata(metadata)), 'INFO'],
          (err) => { if (err) console.error(`[ERROR] Log Save: ${err.message}`); }
        );
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) client.send(JSON.stringify({ event_type, message, metadata: sanitizeMetadata(metadata) }));
        });
      } catch (err) {
        console.error(`[ERROR] WebSocket Message: ${err.message}`);
        logSecurityEvent('WebSocket', 'Message Parse Failure', err.message);
      }
    });
  } catch (err) {
    console.error(`[ERROR] WebSocket Connection: ${err.message}`);
    logSecurityEvent('WebSocket', 'Connection Failure', err.message);
    ws.close();
  }
});

function sanitizeMetadata(metadata) {
  const clean = {};
  for (const [key, value] of Object.entries(metadata || {})) {
    if (typeof value === 'string') {
      clean[key] = value.replace(/[<>]/g, '').replace(/[\n\r\t]/g, ''); // Prevent XSS and control characters
    } else if (typeof value === 'object' && value !== null) {
      clean[key] = sanitizeMetadata(value); // Recursive sanitization
    } else {
      clean[key] = value;
    }
  }
  return clean;
}

function logSecurityEvent(type, message, details) {
  db.run(
    'INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
    [new Date().toISOString(), `security:${type}`, message, JSON.stringify({ details }), 'HIGH'],
    (err) => { if (err) console.error(`[ERROR] Security Log Save: ${err.message}`); }
  );
}

app.post('/mcp/auth', [
  body('token').isString().notEmpty().trim().escape(),
  body('agentId').optional().matches(/^agent_[0-9a-f]{6}$/)
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logSecurityEvent('Auth', 'Validation Failure', JSON.stringify(errors.array()));
    return res.status(400).json({ error: 'Invalid input', details: errors.array() });
  }
  try {
    const { token, agentId } = req.body;
    if (token === 'anonymous') {
      const anonToken = jwt.sign({ scopes: ['vial:read', 'agent:read'], agentId }, secret, { expiresIn: '1h' });
      return res.json({ token: anonToken });
    }
    const decoded = jwt.verify(token, secret);
    if (decoded.scopes.includes('vial:write')) {
      return res.json({ token });
    }
    logSecurityEvent('Auth', 'Unauthorized Access', 'Missing vial:write scope');
    res.status(401).json({ error: 'Invalid token' });
  } catch (err) {
    logSecurityEvent('Auth', 'Token Verification Failure', err.message);
    res.status(401).json({ error: 'Authentication failed' });
  }
});

app.get('/mcp/health', (req, res) => {
  try {
    jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    res.status(200).json({ status: 'ok', agents: Array.from(serviceRegistry.values()).length });
  } catch (err) {
    logSecurityEvent('Health', 'Unauthorized Access', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

app.post('/mcp/vial', [
  body('id').matches(/^vial_[0-9a-f]{6}$/),
  body('code.js').isString().notEmpty().trim().escape(),
  body('training.model').isString().notEmpty(),
  body('training.epochs').isInt({ min: 1 }),
  body('agentId').optional().matches(/^agent_[0-9a-f]{6}$/)
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logSecurityEvent('Vial', 'Validation Failure', JSON.stringify(errors.array()));
    return res.status(400).json({ error: 'Invalid input', details: errors.array() });
  }
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('vial:write')) throw new Error('Missing vial:write scope');
    const { id, code, training, agentId } = req.body;
    if (agentId && !serviceRegistry.has(agentId)) {
      logSecurityEvent('Vial', 'Invalid Agent', `Agent ${agentId} not registered`);
      throw new Error('Agent not registered');
    }
    const vial = {
      id,
      code: JSON.stringify(sanitizeMetadata(code)),
      status: 'running',
      latencyHistory: JSON.stringify([Math.random() * 100]),
      filePath: `/vial/uploads/vial${id}.js`,
      createdAt: new Date().toISOString(),
      codeLength: code.js.length,
      agentId: decoded.agentId || agentId
    };
    db.run(
      'INSERT INTO vials (id, code, status, latencyHistory, filePath, createdAt, codeLength, agentId) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      [vial.id, vial.code, vial.status, vial.latencyHistory, vial.filePath, vial.createdAt, vial.codeLength, vial.agentId],
      (err) => {
        if (err) throw new Error(`Vial Save Error: ${err.message}`);
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({ event_type: 'vial', message: `Vial ${id} created`, metadata: { id, agentId } }));
          }
        });
        res.json(sanitizeMetadata(vial));
      }
    );
  } catch (err) {
    logSecurityEvent('Vial', 'Creation Failure', err.message);
    res.status(500).json({ error: `Vial Creation Error: ${err.message}` });
  }
});

app.get('/mcp/vials', (req, res) => {
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('vial:read')) throw new Error('Missing vial:read scope');
    db.all('SELECT * FROM vials', (err, rows) => {
      if (err) throw new Error(`Vial Fetch Error: ${err.message}`);
      res.json(rows.map(row => sanitizeMetadata({
        id: row.id,
        code: JSON.parse(row.code),
        status: row.status,
        latencyHistory: JSON.parse(row.latencyHistory),
        filePath: row.filePath,
        createdAt: row.createdAt,
        codeLength: row.codeLength,
        agentId: row.agentId
      })));
    });
  } catch (err) {
    logSecurityEvent('Vials', 'Fetch Failure', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

app.post('/mcp/train', [
  body('id').matches(/^vial_[0-9a-f]{6}$/),
  body('input').isString().notEmpty().trim().escape(),
  body('agentId').optional().matches(/^agent_[0-9a-f]{6}$/)
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logSecurityEvent('Train', 'Validation Failure', JSON.stringify(errors.array()));
    return res.status(400).json({ error: 'Invalid input', details: errors.array() });
  }
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('vial:write')) throw new Error('Missing vial:write scope');
    const { id, input, agentId } = req.body;
    if (agentId && !serviceRegistry.has(agentId)) {
      logSecurityEvent('Train', 'Invalid Agent', `Agent ${agentId} not registered`);
      throw new Error('Agent not registered');
    }
    db.get('SELECT * FROM vials WHERE id = ?', [id], (err, row) => {
      if (err || !row) throw new Error(`Vial Not Found: ${err?.message || 'Invalid ID'}`);
      const latency = Math.random() * 100;
      const codeLength = input.length;
      db.run(
        'UPDATE vials SET latencyHistory = ?, code = ?, codeLength = ?, agentId = ? WHERE id = ?',
        [JSON.stringify([...JSON.parse(row.latencyHistory), latency]), JSON.stringify({ js: input }), codeLength, agentId || row.agentId, id],
        (err) => {
          if (err) throw new Error(`Vial Update Error: ${err.message}`);
          wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({ event_type: 'train', message: `Vial ${id} trained`, metadata: { id, latency, agentId } }));
            }
          });
          res.json(sanitizeMetadata({ latency, codeLength }));
        }
      );
    });
  } catch (err) {
    logSecurityEvent('Train', 'Training Failure', err.message);
    res.status(500).json({ error: `Training Error: ${err.message}` });
  }
});

app.post('/mcp/destroy', (req, res) => {
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('vial:write')) throw new Error('Missing vial:write scope');
    db.run('DELETE FROM vials', (err) => {
      if (err) throw new Error(`Vial Destroy Error: ${err.message}`);
      wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(JSON.stringify({ event_type: 'destroy', message: 'All vials destroyed', metadata: {} }));
        }
      });
      res.json({ status: 'ok' });
    });
  } catch (err) {
    logSecurityEvent('Destroy', 'Destroy Failure', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

app.get('/mcp/diagnostics', (req, res) => {
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('vial:read')) throw new Error('Missing vial:read scope');
    const issues = [];
    if (!fs.existsSync(dbPath)) issues.push({ message: 'Database file missing', analysis: 'Check DB_PATH in .env' });
    if (serviceRegistry.size === 0) issues.push({ message: 'No agents registered', analysis: 'Ensure agents use /mcp/register' });
    res.json(sanitizeMetadata({ issues, agents: Array.from(serviceRegistry.values()) }));
  } catch (err) {
    logSecurityEvent('Diagnostics', 'Diagnostics Failure', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

app.post('/mcp/log-sync', [
  body('log').isString().notEmpty().trim().escape()
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logSecurityEvent('LogSync', 'Validation Failure', JSON.stringify(errors.array()));
    return res.status(400).json({ error: 'Invalid input', details: errors.array() });
  }
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('vial:write')) throw new Error('Missing vial:write scope');
    const { log } = req.body;
    const logData = JSON.parse(require('lz-string').decompressFromUTF16(log));
    db.run(
      'INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
      [logData.timestamp, logData.event_type, logData.message, JSON.stringify(sanitizeMetadata(logData.metadata)), logData.urgency],
      (err) => {
        if (err) throw new Error(`Log Sync Error: ${err.message}`);
        res.json({ status: 'ok' });
      }
    );
  } catch (err) {
    logSecurityEvent('LogSync', 'Log Sync Failure', err.message);
    res.status(500).json({ error: `Log Sync Error: ${err.message}` });
  }
});

app.post('/mcp/register', [
  body('id').matches(/^agent_[0-9a-f]{6}$/),
  body('name').isString().notEmpty().trim().escape(),
  body('endpoints').isArray().notEmpty(),
  body('endpoints.*.url').isURL(),
  body('endpoints.*.method').isIn(['GET', 'POST', 'PUT', 'DELETE']),
  body('capabilities.type').isString().notEmpty()
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logSecurityEvent('Register', 'Validation Failure', JSON.stringify(errors.array()));
    return res.status(400).json({ error: 'Invalid input', details: errors.array() });
  }
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('agent:write')) throw new Error('Missing agent:write scope');
    const { id, name, endpoints, capabilities } = req.body;
    const agent = { id, name, endpoints, capabilities, status: 'active', lastPing: new Date().toISOString() };
    db.run(
      'INSERT OR REPLACE INTO agents (id, name, endpoints, capabilities, status, lastPing) VALUES (?, ?, ?, ?, ?, ?)',
      [id, name, JSON.stringify(endpoints), JSON.stringify(sanitizeMetadata(capabilities)), agent.status, agent.lastPing],
      (err) => {
        if (err) throw new Error(`Agent Registration Error: ${err.message}`);
        serviceRegistry.set(id, agent);
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({ event_type: 'agent', message: `Agent ${name} registered`, metadata: { id } }));
          }
        });
        res.json(sanitizeMetadata(agent));
      }
    );
  } catch (err) {
    logSecurityEvent('Register', 'Registration Failure', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

app.get('/mcp/agents', (req, res) => {
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('agent:read')) throw new Error('Missing agent:read scope');
    db.all('SELECT * FROM agents WHERE status = ?', ['active'], (err, rows) => {
      if (err) throw new Error(`Agent Fetch Error: ${err.message}`);
      res.json(rows.map(row => sanitizeMetadata({
        id: row.id,
        name: row.name,
        endpoints: JSON.parse(row.endpoints),
        capabilities: JSON.parse(row.capabilities),
        status: row.status,
        lastPing: row.lastPing
      })));
    });
  } catch (err) {
    logSecurityEvent('Agents', 'Fetch Failure', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

app.post('/mcp/agent/ping', [
  body('agentId').matches(/^agent_[0-9a-f]{6}$/)
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logSecurityEvent('Ping', 'Validation Failure', JSON.stringify(errors.array()));
    return res.status(400).json({ error: 'Invalid input', details: errors.array() });
  }
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('agent:write')) throw new Error('Missing agent:write scope');
    const { agentId } = req.body;
    if (!serviceRegistry.has(agentId)) {
      logSecurityEvent('Ping', 'Invalid Agent', `Agent ${agentId} not registered`);
      throw new Error('Agent not registered');
    }
    const agent = serviceRegistry.get(agentId);
    agent.lastPing = new Date().toISOString();
    agent.status = 'active';
    db.run(
      'UPDATE agents SET lastPing = ?, status = ? WHERE id = ?',
      [agent.lastPing, agent.status, agentId],
      (err) => {
        if (err) throw new Error(`Agent Ping Error: ${err.message}`);
        serviceRegistry.set(agentId, agent);
        res.json(sanitizeMetadata({ status: 'ok', lastPing: agent.lastPing }));
      }
    );
  } catch (err) {
    logSecurityEvent('Ping', 'Ping Failure', err.message);
    res.status(500).json({ error: `Agent Ping Error: ${err.message}` });
  }
});

app.post('/mcp/agent/config', [
  body('agentId').matches(/^agent_[0-9a-f]{6}$/),
  body('config').isObject()
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logSecurityEvent('Config', 'Validation Failure', JSON.stringify(errors.array()));
    return res.status(400).json({ error: 'Invalid input', details: errors.array() });
  }
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('agent:config')) throw new Error('Missing agent:config scope');
    const { agentId, config } = req.body;
    if (!serviceRegistry.has(agentId)) {
      logSecurityEvent('Config', 'Invalid Agent', `Agent ${agentId} not registered`);
      throw new Error('Agent not registered');
    }
    const agent = serviceRegistry.get(agentId);
    agent.capabilities.config = sanitizeMetadata(config);
    db.run(
      'UPDATE agents SET capabilities = ? WHERE id = ?',
      [JSON.stringify(sanitizeMetadata(agent.capabilities)), agentId],
      (err) => {
        if (err) throw new Error(`Agent Config Error: ${err.message}`);
        serviceRegistry.set(agentId, agent);
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({ event_type: 'agent', message: `Agent ${agent.name} reconfigured`, metadata: { agentId, config } }));
          }
        });
        res.json(sanitizeMetadata({ status: 'ok', config }));
      }
    );
  } catch (err) {
    logSecurityEvent('Config', 'Configuration Failure', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

app.get('/mcp/discover', async (req, res) => {
  try {
    const decoded = jwt.verify(req.headers.authorization?.split(' ')[1], secret);
    if (!decoded.scopes.includes('agent:read')) throw new Error('Missing agent:read scope');
    let agents = Array.from(serviceRegistry.values());
    // DNS-based discovery (SRV records)
    try {
      const srvRecords = await dns.resolveSrv('_mcp._tcp.local');
      for (const srv of srvRecords) {
        const agentId = `dns_${srv.name}_${srv.port}`;
        if (!serviceRegistry.has(agentId)) {
          const agent = {
            id: agentId,
            name: srv.name,
            endpoints: [{ url: `http://${srv.name}:${srv.port}/mcp`, method: 'GET' }],
            capabilities: { type: 'dns_discovered', config: {} },
            status: 'active',
            lastPing: new Date().toISOString()
          };
          serviceRegistry.set(agentId, agent);
          db.run(
            'INSERT OR REPLACE INTO agents (id, name, endpoints, capabilities, status, lastPing) VALUES (?, ?, ?, ?, ?, ?)',
            [agent.id, agent.name, JSON.stringify(agent.endpoints), JSON.stringify(agent.capabilities), agent.status, agent.lastPing],
            (err) => { if (err) console.error(`[ERROR] DNS Agent Save: ${err.message}`); }
          );
          agents.push(agent);
        }
      }
    } catch (err) {
      console.error(`[ERROR] DNS Discovery: ${err.message}`);
      logSecurityEvent('Discover', 'DNS Resolution Failure', err.message);
    }
    // Query external AI models
    for (const model of config.ai_models || []) {
      try {
        const response = await axios.get(`${model.endpoint}/mcp/health`, { timeout: 5000 });
        const agentId = `ai_${model.name.toLowerCase()}_${Math.floor(100000 + Math.random() * 900000)}`;
        if (!serviceRegistry.has(agentId)) {
          const agent = {
            id: agentId,
            name: model.name,
            endpoints: [{ url: `${model.endpoint}/mcp`, method: 'POST' }],
            capabilities: { type: 'ai_model', config: model.config },
            status: response.data.status === 'ok' ? 'active' : 'inactive',
            lastPing: new Date().toISOString()
          };
          serviceRegistry.set(agentId, agent);
          db.run(
            'INSERT OR REPLACE INTO agents (id, name, endpoints, capabilities, status, lastPing) VALUES (?, ?, ?, ?, ?, ?)',
            [agent.id, agent.name, JSON.stringify(agent.endpoints), JSON.stringify(agent.capabilities), agent.status, agent.lastPing],
            (err) => { if (err) console.error(`[ERROR] AI Agent Save: ${err.message}`); }
          );
          agents.push(agent);
        }
      } catch (err) {
        console.error(`[ERROR] AI Model ${model.name} Discovery: ${err.message}`);
        logSecurityEvent('Discover', `AI Model ${model.name} Failure`, err.message);
      }
    }
    res.json(sanitizeMetadata(agents));
  } catch (err) {
    logSecurityEvent('Discover', 'Discovery Failure', err.message);
    res.status(401).json({ error: 'Unauthorized' });
  }
});

// Periodic cleanup of stale agents
setInterval(() => {
  const now = Date.now();
  db.all('SELECT id, lastPing FROM agents', (err, rows) => {
    if (err) {
      console.error(`[ERROR] Agent Cleanup: ${err.message}`);
      logSecurityEvent('Cleanup', 'Agent Cleanup Failure', err.message);
      return;
    }
    rows.forEach(row => {
      if (now - new Date(row.lastPing).getTime() > config.agent.ping_interval) {
        db.run('UPDATE agents SET status = ? WHERE id = ?', ['inactive', row.id]);
        serviceRegistry.delete(row.id);
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({ event_type: 'agent', message: `Agent ${row.id} marked inactive`, metadata: { id: row.id } }));
          }
        });
      }
    });
  });
}, config.agent.ping_interval);

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

// Rebuild Instructions: Place in /vial/src/. Install dependencies: `npm install express sqlite3 ws jsonwebtoken dns express-validator express-rate-limit axios`. Ensure /vial/mcp.json and /vial/.env exist. Run with `node server.js`. Check /vial/errorlog.md for issues.
