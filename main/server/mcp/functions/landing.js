// main/server/mcp/functions/landing.js
document.addEventListener('DOMContentLoaded', () => {
  const terminal = document.getElementById('terminal');
  const commandInput = document.getElementById('commandInput');
  const errorDiv = document.getElementById('error');
  const authButtons = document.getElementById('authButtons');
  const importExport = document.getElementById('importExport');
  const dashboardBtn = document.getElementById('dashboardBtn');
  const vialRemoteBtn = document.getElementById('vialRemoteBtn');
  const chatbotBtn = document.getElementById('chatbotBtn');
  const apiKeyBtn = document.getElementById('apiKeyBtn');
  const importFile = document.getElementById('importFile');
  const exportBtn = document.getElementById('exportBtn');
  const troubleshootBtn = document.getElementById('troubleshootBtn');

  const log = (message) => {
    terminal.innerHTML += `${message}\n`;
    terminal.scrollTop = terminal.scrollHeight;
  };

  const fetchStatus = async () => {
    try {
      const response = await fetch('/mcp/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.getSystemMetrics', params: { user_id: 'test_user' }, id: 1 }),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      log(`Status: CPU ${data.result.cpu_usage}%, Memory ${data.result.memory_usage}%, Users ${data.result.active_users}, Balance: ${data.result.balance || 0} WebXOS`);
    } catch (err) {
      log(`Status Error: ${err.message}`);
    }
  };

  const handleCommand = async (command) => {
    log(`> ${command}`);
    try {
      let response;
      const [cmd, ...args] = command.split(' ');
      switch (cmd.toLowerCase()) {
        case '/auth':
          const username = prompt('Enter username:');
          const password = prompt('Enter password:');
          if (!username || !password) throw new Error('Username and password are required');
          response = await fetch('/mcp/auth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
          });
          break;
        case '/help':
          log('Commands: /auth, /status, /help, /troubleshoot');
          return;
        case '/troubleshoot':
          response = await fetch('/mcp/checklist', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
          break;
        default:
          throw new Error(`Unknown command: ${cmd}`);
      }
      const data = await response.json();
      if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);
      log(JSON.stringify(data.result, null, 2));
      if (cmd.toLowerCase() === '/auth' && data.result.redirect) {
        localStorage.setItem('access_token', data.result.access_token);
        authButtons.style.display = 'flex';
        importExport.style.display = 'block';
        log(`Authenticated. Vials: ${JSON.stringify(data.result.vials)}`);
        setInterval(fetchStatus, 5000); // Real-time status
      }
    } catch (err) {
      errorDiv.textContent = `Error: ${err.message}`;
      log(`Error: ${err.message}`);
    }
  };

  dashboardBtn.addEventListener('click', () => window.location.href = '/dashboard');
  vialRemoteBtn.addEventListener('click', () => window.location.href = '/vial');
  chatbotBtn.addEventListener('click', () => window.location.href = '/chatbot');
  apiKeyBtn.addEventListener('click', async () => {
    try {
      const response = await fetch('/mcp/api_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.generateApiKey', params: { user_id: 'test_user' }, id: 2 }),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      log(`API Key: ${data.result}`);
    } catch (err) {
      log(`API Key Error: ${err.message}`);
    }
  });
  exportBtn.addEventListener('click', async () => {
    try {
      const response = await fetch('/mcp/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.exportMd', params: { user_id: 'test_user' }, id: 3 }),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      const blob = new Blob([data.result], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'vial_data.md';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      log(`Export Error: ${err.message}`);
    }
  });
  importFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
      const text = await file.text();
      const response = await fetch('/mcp/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.importMd', params: { user_id: 'test_user', md_content: text }, id: 4 }),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      log(`Imported: ${JSON.stringify(data.result)}`);
    }
  });
  troubleshootBtn.addEventListener('click', () => handleCommand('/troubleshoot'));

  commandInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && commandInput.value.trim()) {
      handleCommand(commandInput.value.trim());
      commandInput.value = '';
    }
  });

  log('Welcome to Vial MCP Terminal! Type /help for commands.');
});
