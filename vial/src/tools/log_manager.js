// src/tools/log_manager.js
exports.syncLog = async (db, LZString, req, res) => {
    try {
        const decompressed = LZString.decompressFromUTF16(req.body.log);
        const logData = JSON.parse(decompressed);
        await new Promise((resolve, reject) => {
            db.run(
                'INSERT INTO logs (timestamp, event_type, message, metadata, urgency) VALUES (?, ?, ?, ?, ?)',
                [logData.timestamp, logData.event_type, logData.message, JSON.stringify(logData.metadata), logData.urgency],
                err => err ? reject(err) : resolve()
            );
        });
        res.json({ message: 'Log synced' });
    } catch (err) {
        console.error(`[LOG_MANAGER] Error: ${err.message}`);
        res.status(500).json({ error: err.message });
    }
};

// Instructions:
// - Syncs client-side logs to server-side SQLite
// - Uses LZ-string for decompression
