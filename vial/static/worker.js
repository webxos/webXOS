/**
 * Web Worker for SQLite database operations
 * Dependencies: /vial/static/sql-wasm.wasm
 * Stores logs and error logs in SQLite for persistence
 * Rebuild: Restore from /vial/src/worker.js if missing
 */
importScripts('/vial/static/sql-wasm.wasm');

let db;

self.onmessage = async (e) => {
  try {
    if (e.data.action === 'init') {
      const SQL = await initSqlJs();
      db = new SQL.Database();
      db.run(`
        CREATE TABLE IF NOT EXISTS logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp TEXT,
          event_type TEXT,
          message TEXT,
          metadata TEXT,
          urgency TEXT
        );
        CREATE TABLE IF NOT EXISTS error_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp TEXT,
          message TEXT,
          analysis TEXT,
          traceback TEXT,
          urgency TEXT
        );
      `);
      self.postMessage({ status: 'initialized' });
    } else if (e.data.action === 'log') {
      const { timestamp, event_type, message, metadata, urgency } = e.data.data;
      db.run(
        'INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
        [timestamp, event_type, message, JSON.stringify(metadata), urgency]
      );
      const logs = db.exec('SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50');
      self.postMessage({ status: 'logs', logs: logs[0]?.values.map(row => ({ timestamp: row[1], message: row[3] })) || [] });
    } else if (e.data.action === 'saveErrorLog') {
      const [timestamp, message, analysis, traceback, urgency] = e.data.content.split('\n').map(line => line.split(': ')[1] || line);
      db.run(
        'INSERT INTO error_logs (timestamp, message, analysis, traceback, urgency) VALUES (?, ?, ?, ?, ?)',
        [timestamp, message, analysis, traceback, urgency]
      );
    }
  } catch (err) {
    self.postMessage({ status: 'error', message: err.message });
  }
};

// Rebuild Instructions: If this file fails to load, restore from /vial/src/worker.js and ensure /vial/static/sql-wasm.wasm is present (download from https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.8.0/dist/sql-wasm.wasm if missing).
