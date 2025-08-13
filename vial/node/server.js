const express = require('express');
const jwt = require('jsonwebtoken');
const dotenv = require('dotenv');
const path = require('path');

dotenv.config({ path: path.resolve(__dirname, '../../.env') });
const app = express();
app.use(express.json());

const SECRET = process.env.JWT_SECRET || 'supersecretkey';

app.post('/auth/verify', (req, res) => {
  try {
    const { token } = req.body;
    const payload = jwt.verify(token, SECRET);
    res.json({ status: 'success', payload });
  } catch (e) {
    console.error(`Auth verification error: ${e.message}`);
    const error = `- **[${new Date().toISOString()}]** Auth verification error: ${e.message}\n`;
    require('fs').appendFileSync(path.resolve(__dirname, '../../vial/errorlog.md'), error);
    res.status(401).json({ error: e.message });
  }
});

app.listen(3000, () => {
  console.log('Node.js auth server running on port 3000');
});
