const fs = require('fs').promises;
const path = require('path');
const sqlite3 = require('sqlite3').verbose();

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
        await logError('Diagnostics DB Initialized', 'SQLite connected in /vial/src/diagnostics.js:20', 'No stack', 'INFO');
    } catch (err) {
        await logError(`Diagnostics DB Error: ${err.message}`, 'Check /vial/src/diagnostics.js:20 or /vial/database.db', err.stack || 'No stack', 'CRITICAL');
        throw err;
    }
}

async function runDiagnostics() {
    try {
        await initDb();
        const issues = [];
        try { await fs.access(dbPath); } catch { issues.push({ message: 'Database file missing', analysis: 'Check /vial/database.db', stack: 'No stack' }); }
        try { await fs.access(errorLogPath); } catch { issues.push({ message: 'Error log file missing', analysis: 'Check /vial/errorlog.md', stack: 'No stack' }); }
        try { await fs.access('/vial/src/agents/nanoGPT.py'); } catch { issues.push({ message: 'NanoGPT script missing', analysis: 'Check /vial/src/agents/nanoGPT.py', stack: 'No stack' }); }
        const staticFiles = ['redaxios.min.js', 'lz-string.min.js', 'mustache.min.js', 'dexie.min.js', 'jwt-decode.min.js', 'sql-wasm.wasm', 'worker.js', 'icon.png', 'manifest.json'];
        for (const file of staticFiles) {
            try { await fs.access(path.join('/vial/static', file)); } catch {
                issues.push({ message: `Static file missing: ${file}`, analysis: `Check /vial/static/${file}`, stack: 'No stack' });
            }
        }
        await logError('Diagnostics Completed', `Found ${issues.length} issues in /vial/src/diagnostics.js:30`, 'No stack', 'INFO');
        return issues;
    } catch (err) {
        await logError(`Diagnostics Error: ${err.message}`, 'Check /vial/src/diagnostics.js:30', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

module.exports = { runDiagnostics };
