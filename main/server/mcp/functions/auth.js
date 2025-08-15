// main/server/mcp/functions/auth.js
document.addEventListener('DOMContentLoaded', () => {
  const authenticateBtn = document.getElementById('authenticateBtn');
  const errorDiv = document.getElementById('error');

  if (authenticateBtn) {
    authenticateBtn.addEventListener('click', async () => {
      try {
        const username = prompt('Enter username:');
        const password = prompt('Enter password:');
        if (!username || !password) throw new Error('Username and password are required');

        const response = await fetch('/mcp/auth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        });
        const data = await response.json();
        if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);

        localStorage.setItem('access_token', data.access_token);
        window.location.href = data.redirect;
      } catch (err) {
        errorDiv.textContent = `Authentication failed: ${err.message}`;
        console.error('Auth error:', err);
      }
    });
  }
});
