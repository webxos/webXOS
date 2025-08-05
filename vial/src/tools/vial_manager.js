// src/tools/vial_manager.js
const fs = require('fs').promises;
const path = require('path');
const vialSchema = require('../../schemas/vial.schema.json');

exports.createVial = async (db, ajv, req, res) => {
    try {
        if (!ajv.validate(vialSchema, req.body)) throw new Error(JSON.stringify(ajv.errors));
        const { id, code, training } = req.body;
        const vial = {
            id,
            code: JSON.stringify(code),
            training: JSON.stringify(training),
            status: 'running',
            latencyHistory: JSON.stringify([Math.random() * 100]),
            filePath: `/uploads/vial${id}.js`,
            createdAt: new Date().toISOString(),
            codeLength: code.js.length
        };
        await new Promise((resolve, reject) => {
            db.run(
                'INSERT INTO vials (id, code, training, status, latencyHistory, filePath, createdAt, codeLength) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                [vial.id, vial.code, vial.training, vial.status, vial.latencyHistory, vial.filePath, vial.createdAt, vial.codeLength],
                err => err ? reject(err) : resolve()
            );
        });
        await fs.writeFile(path.join(__dirname, '../../uploads', `vial${id}.js`), code.js);
        res.json(vial);
    } catch (err) {
        console.error(`[VIAL_MANAGER] Error: ${err.message}`);
        res.status(400).json({ error: err.message });
    }
};

exports.getVials = async (db, req, res) => {
    try {
        const vials = await new Promise((resolve, reject) => {
            db.all('SELECT * FROM vials', (err, rows) => {
                if (err) reject(err);
                else resolve(rows.map(row => ({
                    id: row.id,
                    code: JSON.parse(row.code),
                    training: JSON.parse(row.training),
                    status: row.status,
                    latencyHistory: JSON.parse(row.latencyHistory),
                    filePath: row.filePath,
                    createdAt: row.createdAt,
                    codeLength: row.codeLength
                })));
            });
        });
        res.json(vials);
    } catch (err) {
        console.error(`[VIAL_MANAGER] Error: ${err.message}`);
        res.status(500).json({ error: err.message });
    }
};

exports.destroyAllVials = async (db, req, res) => {
    try {
        await new Promise((resolve, reject) => {
            db.run('DELETE FROM vials', err => err ? reject(err) : resolve());
        });
        const files = await fs.readdir(path.join(__dirname, '../../uploads'));
        for (const file of files) {
            if (file.startsWith('vial')) await fs.unlink(path.join(__dirname, '../../uploads', file));
        }
        res.json({ message: 'All vials destroyed' });
    } catch (err) {
        console.error(`[VIAL_MANAGER] Error: ${err.message}`);
        res.status(500).json({ error: err.message });
    }
};

// Instructions:
// - Manages vials with SQLite
// - Stores code in /uploads
// - Validates with Ajv
