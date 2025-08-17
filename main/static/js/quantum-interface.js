import { MCPClient } from './mcp-client.js';

class QuantumInterface {
  constructor() {
    this.client = new MCPClient();
  }

  async getQuantumState() {
    try {
      const response = await this.client.readResource('quantum://state');
      if (response.error) throw new Error(response.error.message);
      console.log('Quantum state retrieved:', response.result);
      return response.result;
    } catch (error) {
      console.error('Failed to retrieve quantum state:', error.message);
      return { qubits: [], entanglement: 'error' };
    }
  }

  async updateQuantumState(state) {
    try {
      const response = await fetch('https://webxos-mcp-gateway.onrender.com/v1/quantum-link', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({ quantum_state: state })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      console.log('Quantum state updated:', data);
      return data;
    } catch (error) {
      console.error('Failed to update quantum state:', error.message);
      throw error;
    }
  }
}

export { QuantumInterface };
