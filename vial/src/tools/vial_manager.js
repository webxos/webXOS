const sqlite3 = require('sqlite3').verbose();
const fs = require('fs').promises;
const path = require('path');

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
        await logError('Vial Manager DB Initialized', 'SQLite connected in /vial/src/vial_manager.js:20', 'No stack', 'INFO');
    } catch (err) {
        await logError(`Vial Manager DB Error: ${err.message}`, 'Check /vial/src/vial_manager.js:20 or /vial/database.db', err.stack || 'No stack', 'CRITICAL');
        throw err;
    }
}

async function createVial(id, code, filePath) {
    try {
        await initDb();
        const createdAt = new Date().toISOString();
        const codeLength = code.js.length;
        await db.run('INSERT INTO vials (id, status, code, filePath, createdAt, codeLength, latencyHistory) VALUES (?, ?, ?, ?, ?, ?, ?)',
            [id, 'running', JSON.stringify(code), filePath, createdAt, codeLength, JSON.stringify([50])]);
        await fs.writeFile(filePath, code.js);
        await logError(`Vial Created: ${id}`, 'Vial initialized in /vial/src/vial_manager.js:30', 'No stack', 'INFO');
        return { id, status: 'running', code, filePath, createdAt, codeLength, latencyHistory: [50] };
    } catch (err) {
        await logError(`Create Vial Error: ${err.message}`, 'Check /vial/src/vial_manager.js:30 or file permissions', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

async function getVials() {
    try {
        await initDb();
        return new Promise((resolve, reject) => {
            db.all('SELECT * FROM vials', [], (err, rows) => {
                if (err) {
                    logError(`Get Vials Error: ${err.message}`, 'Check /vial/src/vial_manager.js:40', err.stack || 'No stack', 'HIGH');
                    reject(err);
                } else {
                    resolve(rows.map(row => ({
                        id: row.id, status: row.status, code: JSON.parse(row.code), filePath: row.filePath, createdAt: row.createdAt, codeLength: row.codeLength, latencyHistory: JSON.parse(row.latencyHistory)
                    })));
                }
            });
        });
    } catch (err) {
        await logError(`Get Vials Error: ${err.message}`, 'Check /vial/src/vial_manager.js:40', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

async function destroyVials() {
    try {
        await initDb();
        await db.run('DELETE FROM vials');
        await fs.readdir('/vial/uploads').then(files => Promise.all(files.map(file => fs.unlink(path.join('/vial/uploads', file)))));
        await logError('All Vials Destroyed', 'Vials cleared in /vial/src/vial_manager.js:50', 'No stack', 'INFO');
    } catch (err) {
        await logError(`Destroy Vials Error: ${err.message}`, 'Check /vial/src/vial_manager.js:50 or file permissions', err.stack || 'No stack', 'HIGH');
        throw err;
    }
}

module.exports = { createVial, getVials, destroyVials };
