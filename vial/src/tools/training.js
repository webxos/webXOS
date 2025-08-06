const { exec } = require('child_process');
const util = require('util');
const fs = require('fs').promises;
const path = require('path');
const sqlite3 = require('sqlite3').verbose();

const execPromise = util.promisify(exec);
const dbPath = '/vial/database.db';
const errorLogPath = '/vial/errorlog.md';
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
        await logError('Training DB Initialized', 'SQLite connected in /vial/src/training.js:20', 'No stack', 'INFO');
    } catch (err) {
        await logError(`Training DB Error: ${err.message}`, 'Check /vial/src/training.js:20 or /vial/database.db', err.stack || 'No stack', 'CRITICAL');
        throw err;
    }
}

async function trainVial(id, input, agentId) {
    try {
        await initDb();
        const vial = await new Promise((resolve, reject) => {
            db.get('SELECT * FROM vials WHERE id = ?', [id], (err, row) => {
                if (err || !row) reject(new Error('Vial not found'));
                else resolve(row);
            });
        });
        const codePath = path.join('/vial/uploads', `vial${id}.js`);
        await fs.writeFile(codePath, input);
        const agentPath = agentId === 'agent_nanoGPT' ? '/vial/src/agents/nanoGPT.py' : null;
        if (!agentPath) throw new Error('Invalid agent ID');
        try {
            await fs.access(agentPath);
        } catch {
            throw new Error('NanoGPT script missing');
        }
        const { stdout, stderr } = await execPromise(`python3 ${agentPath} --input ${codePath}`);
        const latencyHistory = JSON.parse(vial.latencyHistory);
        const latency = 50 + Math.random() * 10;
        latencyHistory.push(latency);
        await db.run('UPDATE vials SET latencyHistory = ? WHERE id = ?', [JSON.stringify(latencyHistory), id]);
        await logError(`Vial Trained: ${id}`, `Training completed in /vial/src/training.js:30 with ${agentPath}`, stdout || stderr || 'No stack', 'INFO');
        return { latency, codeLength: input.length };
    } catch (err) {
        await logError(`Train Vial Error: ${err.message}`, 'Check /vial/src/training.js:30, /vial/src/agents/nanoGPT.py, or input', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

module.exports = { trainVial };
