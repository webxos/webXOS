const express = require('express');
const cors = require('cors');
const { Isolate } = require('isolated-vm');
const fs = require('fs').promises;

const app = express();
app.use(cors({ origin: '*' })); // Update to GitHub Pages URL in production
app.use(express.json());

app.post('/run', async (req, res) => {
    try {
        const code = req.body.code;
        const isolate = new Isolate({ memoryLimit: 8 });
        const context = isolate.createContextSync();
        const script = await isolate.compileScript(code);
        const result = await script.run(context);
        res.json({ output: result || 'No output' });
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
