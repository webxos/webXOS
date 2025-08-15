// main/server/mcp/functions/terminal.js
document.addEventListener('DOMContentLoaded', () => {
  const terminal = document.getElementById('terminal');
  const commandInput = document.getElementById('commandInput');
  const errorDiv = document.getElementById('error');
  const checklistDiv = document.getElementById('checklist');

  const log = (message) => {
    terminal.innerHTML += `${message}\n`;
    terminal.scrollTop = terminal.scrollHeight;
  };

  const fetchChecklist = async () => {
    try {
      const response = await fetch('/mcp/checklist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      checklistDiv.innerHTML = '<h3>System Checklist:</h3>';
      data.result.missing_files.forEach(file => checklistDiv.innerHTML += `<div class="checklist-item">Missing: ${file}</div>`);
      data.result.fix_steps.forEach(step => checklistDiv.innerHTML += `<div class="checklist-item">Fix: ${step}</div>`);
    } catch (err) {
      checklistDiv.innerHTML = `<div class="checklist-item">Error fetching checklist: ${err.message}</div>`;
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
        case '/status':
          response = await fetch('/mcp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.getSystemMetrics', params: { user_id: 'test_user' }, id: 1 }),
          });
          break;
        case '/help':
          log('Available commands: /auth, /status, /help');
          return;
        default:
          throw new Error(`Unknown command: ${cmd}`);
      }
      const data = await response.json();
      if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);
      log(JSON.stringify(data.result, null, 2));
      if (cmd.toLowerCase() === '/auth' && data.result.redirect) {
        localStorage.setItem('access_token', data.result.access_token);
        window.location.href = data.result.redirect;
      }
    } catch (err) {
      errorDiv.textContent = `Error: ${err.message}`;
      log(`Error: ${err.message}`);
      console.error('Terminal error:', err);
    }
  };

  commandInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && commandInput.value.trim()) {
      handleCommand(commandInput.value.trim());
      commandInput.value = '';
    }
  });

  fetchChecklist();
  log('Welcome to Vial MCP Terminal! Type /help for commands.');
});
