const WS_URL = 'wss://localhost:8000/mcp/notifications';

let websocket = null;

function connectWebSocket(userId) {
  if (websocket) {
    websocket.close();
  }
  
  websocket = new WebSocket(`${WS_URL}?client_id=${userId}`);
  
  websocket.onopen = async () => {
    document.getElementById('websocket-status').innerText = 'Connected';
    console.log('WebSocket connected for user:', userId);
    
    // Sync offline changes
    const pendingImport = localStorage.getItem(`pending_import_${userId}`);
    if (pendingImport) {
      const { markdown, hash } = JSON.parse(pendingImport);
      try {
        const res = await fetch('http://localhost:8000/mcp/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0',
            method: 'wallet.importWallet',
            params: { user_id: userId, markdown, hash },
            id: Math.floor(Math.random() * 1000)
          })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error.message);
        localStorage.removeItem(`pending_import_${userId}`);
        localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: data.result.total_balance }));
        document.getElementById('output').innerText = `Synced offline import: ${data.result.imported_vials.length} vials, new balance: ${data.result.total_balance}`;
      } catch (error) {
        document.getElementById('output').innerText = `Error syncing offline import: ${error.message}`;
      }
    }
    
    const cachedWallet = localStorage.getItem(`wallet_${userId}`);
    if (cachedWallet) {
      const wallet = JSON.parse(cachedWallet);
      try {
        const res = await fetch('http://localhost:8000/mcp/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0',
            method: 'wallet.getVialBalance',
            params: { user_id: userId, vial_id: 'vial1' },
            id: Math.floor(Math.random() * 1000)
          })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error.message);
        if (data.result.balance !== wallet.balance) {
          localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: data.result.balance }));
          document.getElementById('output').innerText = `Synced wallet balance: ${data.result.balance}`;
        }
      } catch (error) {
        document.getElementById('output').innerText = `Error syncing wallet: ${error.message}`;
      }
    }
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
      document.getElementById('vial1-status').innerText = `Running (Balance: ${data.params.balance})`;
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
