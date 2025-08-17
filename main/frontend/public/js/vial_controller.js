const API_URL = 'http://localhost:8000/mcp';

async function fetchVialBalance(userId, vialId) {
  try {
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.getVialBalance',
        params: { user_id: userId, vial_id: vialId },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    document.getElementById('output').innerText = `Error fetching vial balance: ${error.message}`;
    throw error;
  }
}

async function voidVial(userId, vialId) {
  try {
    if (!navigator.onLine) {
      const cachedWallet = localStorage.getItem(`wallet_${userId}`);
      if (!cachedWallet) throw new Error('No wallet data available offline');
      localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: 0 }));
      document.getElementById('output').innerText = `Vial ${vialId} voided offline`;
      updateVialStatus(vialId, 0);
      return;
    }
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.voidVial',
        params: { user_id: userId, vial_id: vialId },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: 0 }));
    document.getElementById('output').innerText = `Vial ${vialId} voided successfully`;
    updateVialStatus(vialId, 0);
  } catch (error) {
    document.getElementById('output').innerText = `Error voiding vial: ${error.message}`;
  }
}

async function troubleshootVial(userId, vialId) {
  try {
    if (!navigator.onLine) {
      const cachedWallet = localStorage.getItem(`wallet_${userId}`);
      if (!cachedWallet) throw new Error('No wallet data available offline');
      const wallet = JSON.parse(cachedWallet);
      document.getElementById('output').innerText = `Offline troubleshoot for ${vialId}: Balance=${wallet.balance}, Active=${wallet.balance > 0}`;
      return;
    }
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.troubleshootVial',
        params: { user_id: userId, vial_id: vialId },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    document.getElementById('output').innerText = `Troubleshoot result for ${vialId}: ${data.result.status}\nDiagnostics: ${JSON.stringify(data.result.diagnostics)}`;
  } catch (error) {
    document.getElementById('output').innerText = `Error troubleshooting vial: ${error.message}`;
  }
}

async function quantumLink(userId) {
  try {
    if (!navigator.onLine) throw new Error('Quantum Link requires online connection');
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.quantumLink',
        params: { user_id: userId },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    document.getElementById('output').innerText = `Quantum Link established: ${data.result.link_id}`;
  } catch (error) {
    document.getElementById('output').innerText = `Error establishing Quantum Link: ${error.message}`;
  }
}

async function exportVials(userId) {
  try {
    if (!navigator.onLine) throw new Error('Export requires online connection');
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.exportVials',
        params: { user_id: userId, vial_id: 'vial1' },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    document.getElementById('output').innerText = `Exported vials: ${data.result.markdown}`;
  } catch (error) {
    document.getElementById('output').innerText = `Error exporting vials: ${error.message}`;
  }
}

async function importWallet(userId, markdown) {
  try {
    const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(markdown));
    const hashHex = Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
    
    if (!navigator.onLine) {
      const pendingOperations = JSON.parse(localStorage.getItem(`pending_operations_${userId}`) || '[]');
      pendingOperations.push({ method: 'importWallet', markdown, hash: hashHex });
      localStorage.setItem(`pending_operations_${userId}`, JSON.stringify(pendingOperations));
      self.postMessage({ action: 'cachePendingImport', data: { user_id: userId, markdown, hash: hashHex } });
      document.getElementById('output').innerText = 'Wallet import queued for next online session';
      document.dispatchEvent(new CustomEvent('sync-progress', { detail: { message: `Pending operations: ${pendingOperations.length}` } }));
      return;
    }
    
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.importWallet',
        params: { user_id: userId, markdown, hash: hashHex },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: data.result.total_balance }));
    document.getElementById('output').innerText = `Imported ${data.result.imported_vials.length} vials, new balance: ${data.result.total_balance}`;
    data.result.imported_vials.forEach(vialId => {
      updateVialStatus(vialId, data.result.total_balance);
    });
  } catch (error) {
    document.getElementById('output').innerText = `Error importing wallet: ${error.message}`;
  }
}

async function mineVial(userId, vialId) {
  try {
    const nonce = Math.floor(Math.random() * 1000000);
    if (!navigator.onLine) {
      const cachedWallet = localStorage.getItem(`wallet_${userId}`);
      if (!cachedWallet) throw new Error('No wallet data available offline');
      const wallet = JSON.parse(cachedWallet);
      const data = `${userId}${vialId}${nonce}`;
      const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(data));
      const hashHex = Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
      let reward = 0;
      if (hashHex.startsWith('00')) {
        reward = 1.0;
        wallet.balance += reward;
        localStorage.setItem(`wallet_${userId}`, JSON.stringify(wallet));
        const pendingOperations = JSON.parse(localStorage.getItem(`pending_operations_${userId}`) || '[]');
        pendingOperations.push({ method: 'mineVial', vial_id: vialId, nonce });
        localStorage.setItem(`pending_operations_${userId}`, JSON.stringify(pendingOperations));
        document.dispatchEvent(new CustomEvent('sync-progress', { detail: { message: `Pending operations: ${pendingOperations.length}` } }));
      }
      document.getElementById('output').innerText = `Offline mining result: Hash=${hashHex}, Reward=${reward}`;
      updateVialStatus(vialId, wallet.balance);
      return;
    }
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.mineVial',
        params: { user_id: userId, vial_id: vialId, nonce },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: data.result.balance }));
    document.getElementById('output').innerText = `Mining result: Hash=${data.result.hash}, Reward=${data.result.reward}`;
    updateVialStatus(vialId, data.result.balance);
  } catch (error) {
    document.getElementById('output').innerText = `Error mining vial: ${error.message}`;
  }
}

async function batchSync(userId) {
  try {
    const pendingOperations = JSON.parse(localStorage.getItem(`pending_operations_${userId}`) || '[]');
    if (!pendingOperations.length) return;
    
    document.dispatchEvent(new CustomEvent('sync-progress', { detail: { message: `Syncing ${pendingOperations.length} operations...` } }));
    
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.batchSync',
        params: { user_id: userId, operations: pendingOperations },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    
    localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: data.result.results[data.result.results.length - 1].total_balance || data.result.results[data.result.results.length - 1].balance }));
    localStorage.removeItem(`pending_operations_${userId}`);
    document.getElementById('output').innerText = `Synced ${data.result.results.length} operations`;
    document.dispatchEvent(new CustomEvent('sync-progress', { detail: { message: 'No pending operations' } }));
    
    data.result.results.forEach(result => {
      if (result.imported_vials) {
        result.imported_vials.forEach(vialId => {
          updateVialStatus(vialId, result.total_balance);
        });
      } else if (result.balance) {
        updateVialStatus('vial1', result.balance);
      }
    });
  } catch (error) {
    document.getElementById('output').innerText = `Error syncing operations: ${error.message}`;
  }
}

function updateVialStatus(vialId, balance) {
  document.getElementById(`${vialId}-status`).innerText = `Running (Balance: ${balance})`;
}

document.addEventListener('auth-success', (event) => {
  const userId = event.detail.user_id;
  ['vial1', 'vial2', 'vial3', 'vial4'].forEach(vialId => {
    fetchVialBalance(userId, vialId).then(data => {
      localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: data.balance }));
      updateVialStatus(vialId, data.balance);
    }).catch(() => {
      const cachedWallet = localStorage.getItem(`wallet_${userId}`);
      if (cachedWallet) {
        const wallet = JSON.parse(cachedWallet);
        updateVialStatus(vialId, wallet.balance);
      }
    });
  });
  if (navigator.onLine) {
    batchSync(userId);
  }
});

document.getElementById('void-btn').addEventListener('click', () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  voidVial(userId, 'vial1');
});

document.getElementById('troubleshoot-btn').addEventListener('click', () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  troubleshootVial(userId, 'vial1');
});

document.getElementById('quantum-link-btn').addEventListener('click', () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  quantumLink(userId);
});

document.getElementById('export-btn').addEventListener('click', () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  exportVials(userId);
});

document.getElementById('import-wallet-btn').addEventListener('click', () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  const markdown = document.getElementById('wallet-import').value;
  importWallet(userId, markdown);
});

document.getElementById('mine-btn').addEventListener('click', () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  mineVial(userId, 'vial1');
});
