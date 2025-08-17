import { MCPClient } from './mcp-client.js';

class Gateway {
  constructor() {
    this.client = new MCPClient();
    this.socket = io('https://webxos-mcp-gateway.onrender.com', { autoConnect: false });
    this.setupSocket();
  }

  setupSocket() {
    this.socket.on('connect', () => console.log('Gateway WebSocket connected'));
    this.socket.on('disconnect', () => console.log('Gateway WebSocket disconnected'));
    this.socket.on('update', (data) => this.handleUpdate(data));
    if (localStorage.getItem('access_token')) this.socket.connect();
  }

  handleUpdate(data) {
    console.log('Gateway update:', data);
    document.dispatchEvent(new CustomEvent('gatewayUpdate', { detail: data }));
  }

  async authenticate(clientId = 'WEBXOS-MOCKKEY', clientSecret = 'MOCKSECRET1234567890') {
    try {
      const response = await fetch('https://webxos-mcp-gateway.onrender.com/v1/oauth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grant_type: 'client_credentials', client_id: clientId, client_secret: clientSecret })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      localStorage.setItem('access_token', data.access_token);
      this.client.token = data.access_token;
      this.socket.connect();
      console.log('Authentication successful');
      return data;
    } catch (error) {
      console.error('Authentication failed:', error.message);
      throw error;
    }
  }

  async getWallet(userId) {
    return await this.client.callTool('get_wallet_info', { user_id: userId });
  }
}

export { Gateway };
