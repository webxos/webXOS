const WS_URL = 'ws://localhost:8000/mcp/notifications';

let websocket = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

function connectWebSocket(clientId) {
  websocket = new WebSocket(`${WS_URL}?client_id=${clientId}`);
  
  websocket.onopen = () => {
    document.getElementById('websocket-status').innerText = 'Connected';
    document.getElementById('quantum-status').innerText = 'Synced';
    reconnectAttempts = 0;
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
    }
  };
  
  websocket.onclose = () => {
    document.getElementById('websocket-status').innerText = 'Disconnected';
    document.getElementById('quantum-status').innerText = 'Disconnected';
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
  };
}

document.addEventListener('auth-success', (event) => {
  const clientId = event.detail.user_id;
  connectWebSocket(clientId);
});

function sendWebSocketMessage(message) {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify(message));
  }
}
