const { OAuth2Client } = require('google-auth-library');
const jwt = require('jsonwebtoken');
const fs = require('fs').promises;

const secret = process.env.OAUTH_CLIENT_SECRET || 'default_secret';
const errorLogPath = '/vial/errorlog.md';

async function logError(message, analysis, stack, urgency) {
    const timestamp = new Date().toISOString();
    const errorMessage = `[${timestamp}] ERROR: ${message}\nAnalysis: ${analysis}\nTraceback: ${stack || 'No stack'}\n---\n`;
    try {
        await fs.appendFile(errorLogPath, errorMessage);
    } catch (err) {
        console.error(`Failed to write to errorlog.md: ${err.message}`);
    }
}

async function authenticate(token) {
    try {
        const client = new OAuth2Client();
        const ticket = await client.verifyIdToken({ idToken: token, audience: secret });
        const payload = ticket.getPayload();
        const newToken = jwt.sign({ sub: payload.sub, exp: Math.floor(Date.now() / 1000) + 3600 }, secret);
        await logError('OAuth Authenticated', 'Token verified in /vial/src/oauth.js:20', 'No stack', 'INFO');
        return newToken;
    } catch (err) {
        await logError(`OAuth Error: ${err.message}`, 'Check /vial/src/oauth.js:20 or /vial/mcp.json:10', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

async function verifyToken(token) {
    try {
        const decoded = jwt.verify(token, secret);
        await logError('Token Verified', 'Token validated in /vial/src/oauth.js:30', 'No stack', 'INFO');
        return decoded;
    } catch (err) {
        await logError(`Token Verification Error: ${err.message}`, 'Check /vial/src/oauth.js:30 or /vial/mcp.json:10', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

module.exports = { authenticate, verifyToken };
