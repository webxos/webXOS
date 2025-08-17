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
    document.getElementById('output').innerText = `Vial ${vialId} voided successfully`;
    updateVialStatus(vialId, 0);
  } catch (error) {
    document.getElementById('output').innerText = `Error voiding vial: ${error.message}`;
  }
}

async function troubleshootVial(userId, vialId) {
  try {
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

function updateVialStatus(vialId, balance) {
  document.getElementById(`${vialId}-status`).innerText = `Running (Balance: ${balance})`;
}

document.addEventListener('auth-success', (event) => {
  const userId = event.detail.user_id;
  ['vial1', 'vial2', 'vial3', 'vial4'].forEach(vialId => {
    fetchVialBalance(userId, vialId).then(data => {
      updateVialStatus(vialId, data.balance);
    });
  });
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
