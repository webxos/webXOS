importScripts('https://cdn.jsdelivr.net/npm/sql.js@1.10.3/dist/sql-wasm.min.js');

let db = null;

self.onmessage = async (e) => {
    const { action, data } = e.data;
    try {
        if (action === 'init') {
            const SQL = await initSqlJs({ locateFile: () => '/static/sql-wasm.wasm' });
            db = new SQL.Database();
            db.run(`
                CREATE TABLE logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    message TEXT,
                    metadata TEXT,
                    urgency TEXT
                );
            `);
            self.postMessage({ status: 'initialized' });
        } else if (action === 'log') {
            db.run('INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
                [data.timestamp, data.event_type, data.message, JSON.stringify(data.metadata), data.urgency]);
            self.postMessage({ status: 'logged' });
        } else if (action === 'getLogs') {
            const results = db.exec('SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50');
            const logs = results.length ? results[0].values.map(row => ({
                id: row[0],
                timestamp: row[1],
                event_type: row[2],
                message: row[3],
                metadata: JSON.parse(row[4]),
                urgency: row[5]
            })) : [];
            self.postMessage({ status: 'logs', logs });
        }
    } catch (err) {
        self.postMessage({ status: 'error', message: err.message });
    }
};
