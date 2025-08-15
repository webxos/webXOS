// main/server/mcp/functions/quantum.js
import { callTool } from './mcp.js';

export async function simulateQuantumCircuit(numQubits, gates, numShots = 1024) {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('apiKey')}`,
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.simulateCircuit',
        params: {
          circuit_data: {
            num_qubits: numQubits,
            gates: gates
          },
          num_shots: numShots
        },
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    throw new Error(`Quantum simulation failed: ${error.message}`);
  }
}

export async function getCircuitResult(circuitId) {
  try {
    const response = await callTool('get_circuit_result', { circuit_id: circuitId });
    return response;
  } catch (error) {
    throw new Error(`Failed to retrieve circuit result: ${error.message}`);
  }
}
