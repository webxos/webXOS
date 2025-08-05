// src/db/sqlite.js
const sqlite3 = require('sqlite3').verbose();

module.exports = new sqlite3.Database('./uploads/vial.db', (err) => {
    if (err) console.error(`[SQLITE] Error: ${err.message}`);
});

// Instructions:
// - SQLite connection
// - Database: ./Uploads/vial.db
