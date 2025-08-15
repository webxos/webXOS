// main/server/mcp/functions/quantum.js
async function simulateQuantumCircuit(vialId, circuitData) {
  const token = localStorage.getItem('apiKey');
  const userId = localStorage.getItem('userId');
  if (!token || !userId) throw new Error('Not authenticated');
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/quantum/simulate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ vial_id: vialId, circuit_data: circuitData, user_id: userId })
  });
  if (!response.ok) throw new Error(`Quantum simulation failed: ${await response.text()}`);
  return await response.json();
}

async function getQuantumResults(vialId) {
  const token = localStorage.getItem('apiKey');
  const userId = localStorage.getItem('userId');
  if (!token || !userId) throw new Error('Not authenticated');
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/quantum/results/${vialId}`, {
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (!response.ok) throw new Error(`Failed to fetch quantum results: ${await response.text()}`);
  return await response.json();
}

export { simulateQuantumCircuit, getQuantumResults };
