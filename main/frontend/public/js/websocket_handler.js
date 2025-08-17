const WS_URL = 'ws://localhost:8000/mcp/notifications';

let websocket = null;

function connectWebSocket(userId) {
  if (websocket) {
    websocket.close();
  }
  
  websocket = new WebSocket(`${WS_URL}?client_id=${userId}`);
  
  websocket.onopen = () => {
    document.getElementById('websocket-status').innerText = 'Connected';
    console.log('WebSocket connected for user:', userId);
  };
  
  websocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received WebSocket message:', data);
    
    if (data.method === 'wallet.voidVial') {
      document.getElementById('output').innerText = `Notification: Vial ${data.params.vial_id} voided`;
      document.getElementById(`${data.params.vial_id}-status`).innerText = 'Stopped (Balance: 0)';
    } else if (data.method === 'wallet.troubleshootVial') {
      document.getElementById('output').innerText = `Notification: Troubleshoot ${data.params.vial_id}: ${data.params.status}\nDiagnostics: ${JSON.stringify(data.params.diagnostics)}`;
    } else if (data.method === 'wallet.quantumLink') {
      document.getElementById('output').innerText = `Notification: Quantum Link established: ${data.params.link_id}`;
    } else if (data.method === 'wallet.importWallet') {
      document.getElementById('output').innerText = `Notification: Imported ${data.params.imported_vials.length} vials, new balance: ${data.params.total_balance}`;
      data.params.imported_vials.forEach(vialId => {
        document.getElementById(`${vialId}-status`).innerText = `Running (Balance: ${data.params.total_balance})`;
      });
    } else if (data.method === 'wallet.mineVial') {
      document.getElementById('output').innerText = `Notification: Mining result: Hash=${data.params.hash}, Reward=${data.params.reward}`;
    } else if (data.method === 'claude.executeCode') {
      document.getElementById('output').innerText = `Notification: Claude code executed: ${data.params.output}`;
    }
  };
  
  websocket.onclose = () => {
    document.getElementById('websocket-status').innerText = 'Disconnected';
    console.log('WebSocket disconnected, attempting to reconnect...');
    setTimeout(() => connectWebSocket(userId), 5000);
  };
  
  websocket.onerror = (error) => {
    console.error('WebSocket error:', error);
    document.getElementById('websocket-status').innerText = 'Error';
  };
}

document.addEventListener('auth-success', (event) => {
  const userId = event.detail.user_id;
  connectWebSocket(userId);
});
