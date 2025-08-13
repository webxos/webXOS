const express = require('express');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const app = express();
const port = process.env.PORT || 3001;

app.use(express.json());

// OAuth micro-gateway
app.post('/api/oauth/token', async (req, res) => {
  try {
    const { code } = req.body;
    const response = await axios.post(
      'https://github.com/login/oauth/access_token',
      {
        client_id: process.env.OAUTH_CLIENT_ID,
        client_secret: process.env.OAUTH_CLIENT_SECRET,
        code
      },
      { headers: { Accept: 'application/json' } }
    );
    res.json(response.data);
  } catch (error) {
    console.error(`OAuth error: ${error.message}`);
    res.status(500).json({ error: 'OAuth token retrieval failed' });
  }
});

// Git operations
app.post('/api/git', async (req, res) => {
  const { user_id, command, repo_url, token } = req.body;
  const allowed_commands = ['git clone', 'git commit', 'git push', 'git pull', 'git branch', 'git merge'];
  if (!allowed_commands.some(cmd => command.startsWith(cmd))) {
    return res.status(400).json({ error: 'Invalid Git command' });
  }
  
  try {
    const { execSync } = require('child_process');
    const result = execSync(command, { cwd: '/tmp/repo' });
    res.json({ status: 'success', output: result.toString() });
  } catch (error) {
    console.error(`Git command error: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Node.js micro-gateway running on port ${port}`);
});
