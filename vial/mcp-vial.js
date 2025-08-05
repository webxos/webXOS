const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs').promises;
const path = require('path');
const jwt = require('jsonwebtoken');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: '*',
        methods: ['GET', 'POST']
    }
});

const SECRET_KEY = 'your-secret-key';
const VIAL_LIMIT = 4;
let activeVials = [];

app.use(express.json());
app.use('/static', express.static(path.join(__dirname, 'static')));
app.use('/webxos/vial/lab', express.static(path.join(__dirname, 'webxos/vial/lab')));

app.get('/', async (req, res) => {
    try {
        const html = await fs.readFile(path.join(__dirname, 'Vial.html'), 'utf8');
        res.send(html);
    } catch (err) {
        console.error(`[MCP-VIAL] Error serving Vial.html: ${err.message}`);
        res.status(500).send('Server Error');
    }
});

app.post('/api/auth/login', (req, res) => {
    try {
        const { username, password } = req.body;
        if (username === 'user' && password === 'pass') {
            const token = jwt.sign({ username }, SECRET_KEY, { expiresIn: '1h' });
            res.json({ token });
        } else {
            throw new Error('Invalid credentials');
        }
    } catch (err) {
        console.error(`[MCP-VIAL] Auth Error: ${err.message}`);
        res.status(401).json({ message: 'Authentication failed', analysis: 'Check username or password.' });
    }
});

app.post('/api/input', async (req, res) => {
    try {
        const authHeader = req.headers.authorization;
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            throw new Error('No token provided');
        }
        const token = authHeader.split(' ')[1];
        jwt.verify(token, SECRET_KEY);
        res.json({ message: 'File uploaded successfully' });
    } catch (err) {
        console.error(`[MCP-VIAL] File Upload Error: ${err.message}`);
        res.status(401).json({ message: `File upload failed: ${err.message}`, analysis: 'Check token or file format.' });
    }
});

async function generateVialFile(vialId, code) {
    try {
        const labDir = path.join(__dirname, 'webxos/vial/lab');
        await fs.mkdir(labDir, { recursive: true });
        
        const vialScript = `
/**
 * Vial Agent Script for ${vialId}
 * Generated automatically by Vial MCP Server
 * @module ${vialId}
 */
class VialAgent {
    constructor(id) {
        this.id = id;
        this.status = 'running';
        this.createdAt = new Date().toISOString();
        this.code = \`${code || 'default'}\`;
    }

    async init() {
        try {
            console.log(\`[VIAL \${this.id}] Initializing vial agent\`);
            // Placeholder for WebXOS integration
            return { status: this.status, latency: Math.random() * 100 };
        } catch (err) {
            console.error(\`[VIAL \${this.id}] Init Error: \${err.message}\`);
            throw err;
        }
    }

    async process() {
        try {
            console.log(\`[VIAL \${this.id}] Processing vial agent\`);
            // Simulate processing with WebXOS
            return { result: Math.random(), latency: Math.random() * 100 };
        } catch (err) {
            console.error(\`[VIAL \${this.id}] Process Error: \${err.message}\`);
            throw err;
        }
    }
}

export default new VialAgent('${vialId}');
`;

        const filePath = path.join(labDir, `vial${vialId}.js`);
        await fs.writeFile(filePath, vialScript);
        console.log(`[MCP-VIAL] Generated ${filePath}`);
        return filePath;
    } catch (err) {
        console.error(`[MCP-VIAL] File Generation Error: ${err.message}`);
        throw err;
    }
}

io.on('connection', (socket) => {
    console.log('[MCP-VIAL] Client connected');
    
    socket.on('create-vial', async (data) => {
        try {
            if (activeVials.length >= VIAL_LIMIT) {
                socket.emit('server-error', {
                    message: 'Maximum 4 vials allowed per session',
                    analysis: 'VOID existing vials to create new ones.'
                });
                return;
            }

            const vialId = Math.floor(100000 + Math.random() * 900000).toString();
            const filePath = await generateVialFile(vialId, data.code);
            activeVials.push({ id: vialId, status: 'running', createdAt: new Date(), filePath });

            socket.emit('vial-created', {
                vialId,
                latency: Math.random() * 100,
                code: data.code
            });

            console.log(`[MCP-VIAL] Vial ${vialId} created at ${filePath}`);
        } catch (err) {
            console.error(`[MCP-VIAL] Create Vial Error: ${err.message}`);
            socket.emit('server-error', {
                message: `Failed to create vial: ${err.message}`,
                analysis: 'Check server file system permissions or WebXOS integration.'
            });
        }
    });

    socket.on('check-vials', () => {
        try {
            activeVials.forEach(vial => {
                socket.emit('vial-status', {
                    vial: vial.id,
                    status: vial.status,
                    latency: Math.random() * 100
                });
            });
        } catch (err) {
            console.error(`[MCP-VIAL] Check Vials Error: ${err.message}`);
            socket.emit('server-error', {
                message: `Failed to check vials: ${err.message}`,
                analysis: 'Check server state or vial data.'
            });
        }
    });

    socket.on('troubleshoot-vials', () => {
        try {
            activeVials.forEach(vial => {
                socket.emit('vial-status', {
                    vial: vial.id,
                    status: vial.status,
                    latency: Math.random() * 100
                });
            });
        } catch (err) {
            console.error(`[MCP-VIAL] Troubleshoot Error: ${err.message}`);
            socket.emit('server-error', {
                message: `Troubleshoot failed: ${err.message}`,
                analysis: 'Check server logs or vial status.'
            });
        }
    });

    socket.on('void-vials', () => {
        try {
            activeVials = [];
            socket.emit('server-error', {
                message: 'All vials destroyed',
                analysis: 'Vial state reset successfully.'
            });
            console.log('[MCP-VIAL] All vials destroyed');
        } catch (err) {
            console.error(`[MCP-VIAL] Void Vials Error: ${err.message}`);
            socket.emit('server-error', {
                message: `Failed to void vials: ${err.message}`,
                analysis: 'Check server state.'
            });
        }
    });

    socket.on('disconnect', () => {
        console.log('[MCP-VIAL] Client disconnected');
    });
});

const PORT = process.env.PORT || 8080;
server.listen(PORT, async () => {
    try {
        await fs.mkdir(path.join(__dirname, 'webxos/vial/lab'), { recursive: true });
        console.log(`[MCP-VIAL] Server running on port ${PORT}`);
    } catch (err) {
        console.error(`[MCP-VIAL] Server Start Error: ${err.message}`);
    }
});
