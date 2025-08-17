const WS_URL = 'ws://localhost:8000/mcp/notifications';

let websocket = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

function connectWebSocket(clientId) {
  websocket = new WebSocket(`${WS_URL}?client_id=${clientId}`);
  
  websocket.onopen = () => {
    document.getElementById('websocket-status').innerText = 'Connected';
    document.getElementById('quantum-status').innerText = 'Synced';
    document.getElementById('wallet-status').innerText = 'Enabled';
    reconnectAttempts = 0;
    // Trigger batch sync on reconnect
    const event = new CustomEvent('auth-success', { detail: { user_id: clientId } });
    document.dispatchEvent(event);
  };
  
  websocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.method.includes('wallet')) {
      document.getElementById('output').innerText = `Wallet update: ${JSON.stringify(data.params)}`;
      if (data.params.quantum_state) {
        document.getElementById('quantum-status').innerText = data.params.quantum_state.entanglement;
      }
      if (data.params.vial_id && data.params.balance) {
        document.getElementById(`${data.params.vial_id}-status`).innerText = `Running (Balance: ${data.params.balance})`;
      }
      if (data.params.total_balance) {
        document.getElementById('balance').innerText = `${data.params.total_balance} $WEBXOS`;
      }
      if (data.params.commit_hash) {
        document.getElementById('output').innerText = `Git push successful: Commit ${data.params.commit_hash}`;
      }
    }
  };
  
  websocket.onclose = () => {
    document.getElementById('websocket-status').innerText = 'Disconnected';
    document.getElementById('quantum-status').innerText = 'Disconnected';
    document.getElementById('wallet-status').innerText = 'Disabled';
    if (reconnectAttempts < maxReconnectAttempts) {
      setTimeout(() => {
        reconnectAttempts++;
        connectWebSocket(clientId);
      }, 5000);
    } else {
      document.getElementById('output').innerText = 'WebSocket connection failed after max retries';
    }
  };
  
  websocket.onerror = (error) => {
    document.getElementById('output').innerText = `WebSocket error: ${error}`;
    document.getElementById('wallet-status').innerText = 'Disabled';
  };
}

function sendWebSocketMessage(message) {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify(message));
  } else {
    document.getElementById('output').innerText = 'Error: Wallet disabled due to WebSocket disconnection';
  }
}

document.addEventListener('auth-success', (event) => {
  const clientId = event.detail.user_id;
  connectWebSocket(clientId);
});
