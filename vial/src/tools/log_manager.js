const sqlite3 = require('sqlite3').verbose();
const fs = require('fs').promises;

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
        await logError('Log Manager DB Initialized', 'SQLite connected in /vial/src/log_manager.js:20', 'No stack', 'INFO');
    } catch (err) {
        await logError(`Log Manager DB Error: ${err.message}`, 'Check /vial/src/log_manager.js:20 or /vial/database.db', err.stack || 'No stack', 'CRITICAL');
        throw err;
    }
}

async function saveLog(timestamp, event_type, message, metadata, urgency) {
    try {
        await initDb();
        await db.run('INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
            [timestamp, event_type, message, JSON.stringify(metadata), urgency]);
        await logError(`Log Saved: ${event_type}`, 'Log saved in /vial/src/log_manager.js:30', 'No stack', 'INFO');
    } catch (err) {
        await logError(`Save Log Error: ${err.message}`, 'Check /vial/src/log_manager.js:30 or /vial/database.db', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

async function getLogs() {
    try {
        await initDb();
        return new Promise((resolve, reject) => {
            db.all('SELECT * FROM logs', [], (err, rows) => {
                if (err) {
                    logError(`Get Logs Error: ${err.message}`, 'Check /vial/src/log_manager.js:40', err.stack || 'No stack', 'HIGH');
                    reject(err);
                } else {
                    resolve(rows.map(row => ({
                        id: row.id, timestamp: row.timestamp, event_type: row.event_type, message: row.message, metadata: JSON.parse(row.metadata), urgency: row.urgency
                    })));
                }
            });
        });
    } catch (err) {
        await logError(`Get Logs Error: ${err.message}`, 'Check /vial/src/log_manager.js:40', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

module.exports = { saveLog, getLogs };
