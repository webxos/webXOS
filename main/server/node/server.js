/**
 * Node.js fallback authentication server for Vial MCP.
 */
const express = require('express');
const jwt = require('jsonwebtoken');
const { verifyApiKey } = require('./src/auth_verification');
const { syncNetwork } = require('./src/network_sync');

const app = express();
app.use(express.json());

const JWT_SECRET = process.env.JWT_SECRET_KEY || 'your_jwt_secret_key';

app.post('/auth/login', async (req, res) => {
    const { api_key, wallet_id } = req.body;
    try {
        if (!verifyApiKey(api_key, wallet_id)) {
            return res.status(401).json({ error: 'Invalid API key or wallet ID' });
        }
        const token = jwt.sign({ wallet_id }, JWT_SECRET, { expiresIn: '1h' });
        await syncNetwork({ endpoint: '/auth/login', status: 'success', wallet_id });
        res.json({ access_token: token, expires_in: 3600 });
    } catch (error) {
        await syncNetwork({ endpoint: '/auth/login', status: 'error', error: error.message });
        res.status(500).json({ error: 'Authentication failed' });
    }
});

app.listen(8080, () => {
    console.log('Fallback auth server running on port 8080');
});
