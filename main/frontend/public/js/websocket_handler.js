let ws = null;

function initWebSocket(userId) {
  if (!userId || userId === 'Not logged in') {
    document.getElementById('websocket-status').innerText = 'Disconnected (Please authenticate)';
    return;
  }

  ws = new WebSocket(`ws://localhost:8000/mcp/notifications?client_id=${userId}`);

  ws.onopen = () => {
    document.getElementById('websocket-status').innerText = 'Connected';
    console.log('WebSocket connected for user:', userId);
  };

  ws.onmessage = (event) => {
    const notification = JSON.parse(event.data);
    if (notification.method === 'claude.executionComplete') {
      const { output, error } = notification.params;
      document.getElementById('output').innerText = `Claude Output: ${output || 'No output'}${error ? '\nError: ' + error : ''}`;
    }
  };

  ws.onclose = () => {
    document.getElementById('websocket-status').innerText = 'Disconnected';
    console.log('WebSocket disconnected');
    setTimeout(() => initWebSocket(userId), 5000); // Reconnect after 5s
  };

  ws.onerror = (error) => {
    document.getElementById('websocket-status').innerText = 'Error';
    console.error('WebSocket error:', error);
  };
}

// Update auth_handler.js to call initWebSocket on successful login
document.addEventListener('auth-success', (event) => {
  initWebSocket(event.detail.user_id);
});
