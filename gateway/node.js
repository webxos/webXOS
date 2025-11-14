const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const fs = require('fs').promises;
const app = express();

app.use(cors({ origin: '*' }));
app.use(express.json());

app.post('/run', async (req, res) => {
    try {
        const code = req.body.code;
        // Write code to a temp file for execution (basic, insecure example)
        await fs.writeFile('temp.js', code);
        exec('node temp.js', (err, stdout, stderr) => {
            if (err) return res.json({ output: stderr });
            res.json({ output: stdout });
        });
    } catch (error) {
        res.status(500).json({ output: `Error: ${error.message}` });
    }
});

app.post('/save', async (req, res) => {
    try {
        await fs.writeFile('saved.js', req.body.code);
        res.json({ message: 'Code saved' });
    } catch (error) {
        res.status(500).json({ message: `Save error: ${error.message}` });
    }
});

app.listen(3000, () => console.log('Node.js on port 3000'));
