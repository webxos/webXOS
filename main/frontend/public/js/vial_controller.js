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
  document.getElementById('output').innerText = 'Void operation not implemented';
});

document.getElementById('troubleshoot-btn').addEventListener('click', () => {
  document.getElementById('output').innerText = 'Troubleshooting not implemented';
});

document.getElementById('quantum-link-btn').addEventListener('click', () => {
  document.getElementById('output').innerText = 'Quantum Link not implemented';
});

document.getElementById('export-btn').addEventListener('click', () => {
  document.getElementById('output').innerText = 'Export not implemented';
});
