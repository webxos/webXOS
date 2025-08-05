const { Server } = require('socket.io');
const http = require('http');
const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const jwt = require('jsonwebtoken');
const { MongoClient } = require('mongodb');
const { loadPyodide } = require('pyodide');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*' },
    maxHttpBufferSize: 1e7
});

app.use(bodyParser.json());
app.use(express.static('static'));

const upload = multer({ storage: multer.memoryStorage() });
const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret';
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/vial_mcp';
let db = null;
let pyodide = null;

async function initMongoDB() {
    try {
        const client = new MongoClient(MONGODB_URI);
        await client.connect();
        db = client.db('vial_mcp');
        console.log('[VIAL] Connected to MongoDB');
    } catch (err) {
        console.error('[VIAL] MongoDB Connection Error:', err);
    }
}

async function initPyodide() {
    try {
        pyodide = await loadPyodide({ indexURL: 'https://cdn.pyodide.org/v0.26.2/full/' });
        await pyodide.loadPackage(['micropip']);
        await pyodide.runPythonAsync(`
            import micropip
            await micropip.install('numpy')
        `);
        console.log('[VIAL] Pyodide initialized with numpy');
    } catch (err) {
        console.error('[VIAL] Pyodide Init Error:', err);
    }
}

function logError(message, error) {
    console.error(`[VIAL] ${message}: ${error.message}\nStack: ${error.stack}`);
}

function generateVialId() {
    return `vial${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function verifyToken(req, res, next) {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'No token provided' });
    try {
        req.user = jwt.verify(token, JWT_SECRET);
        next();
    } catch (err) {
        res.status(403).json({ error: 'Invalid token' });
    }
}

app.post('/api/auth/login', (req, res) => {
    try {
        const { username, password } = req.body;
        if (username === 'user' && password === 'pass') {
            const token = jwt.sign({ username }, JWT_SECRET, { expiresIn: '1h' });
            res.json({ token });
        } else {
            res.status(401).json({ error: 'Invalid credentials' });
        }
    } catch (err) {
        logError('Login Error', err);
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/input', verifyToken, upload.single('file'), async (req, res) => {
    try {
        const { code, vialId } = req.body;
        const file = req.file;
        if (!code && !file) {
            return res.status(400).json({ error: 'Missing code or file' });
        }
        if (!vialId) {
            return res.status(400).json({ error: 'Missing vialId' });
        }
        const content = file ? file.buffer.toString() : code;
        console.log(`[VIAL] Received code for ${vialId}: ${content.substring(0, 50)}...`);
        if (db) {
            await db.collection('vials').updateOne(
                { name: vialId },
                { $set: { code: content, updatedAt: new Date() } },
                { upsert: true }
            );
        }
        if (content.endsWith('.py') && pyodide) {
            try {
                const pyResult = await pyodide.runPythonAsync(content);
                res.json({ message: `Code executed for ${vialId}`, vialId, pyResult });
            } catch (pyErr) {
                res.status(400).json({ error: pyErr.message, analysis: 'Check Python syntax' });
            }
        } else {
            res.json({ message: `Code received for ${vialId}`, vialId });
        }
    } catch (err) {
        logError('API Input Error', err);
        res.status(500).json({ error: err.message, analysis: 'Check server logs.' });
    }
});

app.get('/api/output/:vialId', verifyToken, async (req, res) => {
    try {
        const { vialId } = req.params;
        let vial = null;
        if (db) {
            vial = await db.collection('vials').findOne({ name: vialId });
        }
        if (!vial) {
            return res.status(404).json({ error: `Vial ${vialId} not found` });
        }
        res.json({ vialId, output: vial.output || `Simulated output for ${vialId}` });
    } catch (err) {
        logError('API Output Error', err);
        res.status(500).json({ error: err.message, analysis: 'Check server logs.' });
    }
});

io.on('connection', (socket) => {
    console.log('[VIAL] Client connected:', socket.id);

    socket.on('create-vial', async (data) => {
        try {
            if (!socket.handshake.auth.token) {
                socket.emit('server-error', { message: 'Authentication required', analysis: 'Provide valid JWT token' });
                return;
            }
            jwt.verify(socket.handshake.auth.token, JWT_SECRET);
            const vialId = generateVialId();
            const newVial = { name: vialId, status: 'running', startTime: new Date(), code: data.code || 'default', output: '' };
            if (db) {
                await db.collection('vials').insertOne(newVial);
            }
            const latency = Math.random() * 100;
            socket.emit('vial-created', { vialId, message: `Vial ${vialId} created.`, latency });
            socket.emit('vial-status', {
                vial: vialId,
                status: 'started',
                latency,
                timestamp: newVial.startTime.toISOString()
            });
            console.log(`[VIAL] ${vialId} created at ${newVial.startTime.toISOString()} with code: ${newVial.code.substring(0, 50)}...`);
            if (data.code.endsWith('.py') && pyodide) {
                try {
                    const pyResult = await pyodide.runPythonAsync(data.code);
                    newVial.output = pyResult;
                    if (db) {
                        await db.collection('vials').updateOne(
                            { name: vialId },
                            { $set: { output: pyResult } }
                        );
                    }
                } catch (pyErr) {
                    socket.emit('server-error', { message: pyErr.message, analysis: 'Check Python syntax' });
                }
            }
        } catch (err) {
            socket.emit('server-error', { 
                message: err.message, 
                analysis: 'Failed to create vial. Check server logs and ensure WebXOS tools are accessible at webxos.netlify.app.' 
            });
            logError('Create Vial Error', err);
        }
    });

    socket.on('troubleshoot-vials', async () => {
        try {
            if (!socket.handshake.auth.token) {
                socket.emit('server-error', { message: 'Authentication required', analysis: 'Provide valid JWT token' });
                return;
            }
            let vials = [];
            if (db) {
                vials = await db.collection('vials').find({}).toArray();
            }
            if (vials.length === 0) {
                socket.emit('server-error', { message: 'No vials found.', analysis: 'Create a vial first.' });
                return;
            }
            vials.forEach(vial => {
                const errorChance = Math.random();
                let analysis = 'No issues detected.';
                if (errorChance > 0.7) {
                    analysis = `Potential issue in ${vial.name}: Check code syntax or WebXOS integration at webxos.netlify.app.`;
                } else if (errorChance > 0.4) {
                    analysis = `Warning in ${vial.name}: Possible performance bottleneck in code execution.`;
                }
                socket.emit('vial-status', {
                    vial: vial.name,
                    status: `troubleshooted: ${analysis}`,
                    latency: vial.status === 'running' ? Math.random() * 100 : 0,
                    timestamp: new Date().toISOString()
                });
                console.log(`[VIAL] ${vial.name} troubleshooted: ${analysis}`);
            });
        } catch (err) {
            socket.emit('server-error', { 
                message: err.message, 
                analysis: 'Failed to troubleshoot vials. Check server logs and WebXOS integration.' 
            });
            logError('Troubleshoot Vials Error', err);
        }
    });

    socket.on('check-vials', async () => {
        try {
            if (!socket.handshake.auth.token) {
                socket.emit('server-error', { message: 'Authentication required', analysis: 'Provide valid JWT token' });
                return;
            }
            let vials = [];
            if (db) {
                vials = await db.collection('vials').find({}).toArray();
            }
            if (vials.length === 0) {
                socket.emit('server-error', { message: 'No vials found.', analysis: 'Create a vial first.' });
                return;
            }
            vials.forEach(vial => {
                const latency = Math.random() * 100;
                socket.emit('vial-status', {
                    vial: vial.name,
                    status: vial.status,
                    latency: vial.status === 'running' ? latency : 0,
                    timestamp: new Date().toISOString()
                });
                console.log(`[VIAL] ${vial.name} checked: ${vial.status}, Latency: ${latency.toFixed(2)}ms`);
            });
        } catch (err) {
            socket.emit('server-error', { 
                message: err.message, 
                analysis: 'Failed to check vials. Check server logs.' 
            });
            logError('Check Vials Error', err);
        }
    });

    socket.on('void-vials', async () => {
        try {
            if (!socket.handshake.auth.token) {
                socket.emit('server-error', { message: 'Authentication required', analysis: 'Provide valid JWT token' });
                return;
            }
            let vials = [];
            if (db) {
                vials = await db.collection('vials').find({}).toArray();
                await db.collection('vials').deleteMany({});
            }
            vials.forEach(vial => {
                if (vial.status === 'running') {
                    const runTime = Math.round((new Date() - vial.startTime) / 1000);
                    socket.emit('vial-status', {
                        vial: vial.name,
                        status: `destroyed, ran for ${runTime}s`,
                        latency: 0,
                        timestamp: new Date().toISOString()
                    });
                    console.log(`[VIAL] ${vial.name} destroyed, ran for ${runTime}s`);
                }
            });
            socket.emit('server-error', { 
                message: 'All vials destroyed.', 
                analysis: 'System reset successfully.' 
            });
        } catch (err) {
            socket.emit('server-error', { 
                message: err.message, 
                analysis: 'Failed to void vials. Check server logs.' 
            });
            logError('Void Vials Error', err);
        }
    });

    socket.on('disconnect', () => {
        console.log('[VIAL] Client disconnected:', socket.id);
    });
});

server.listen(8080, async () => {
    console.log('[VIAL] Server running on ws://localhost:8080');
    await initMongoDB();
    await initPyodide();
});

process.on('uncaughtException', (err) => {
    console.error('[VIAL] Uncaught Exception:', err.message, '\nStack:', err.stack);
    io.emit('server-error', { message: 'Server encountered an unexpected error.', analysis: 'Check server logs.' });
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('[VIAL] Unhandled Rejection at:', promise, 'Reason:', reason);
    io.emit('server-error', { message: 'Server encountered an unexpected error.', analysis: 'Check server logs.' });
});
