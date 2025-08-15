// main/server/mcp/functions/mcp_inspector.js
const fetchWithTimeout = async (url, options, timeout = 5000) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  const response = await fetch(url, { ...options, signal: controller.signal });
  clearTimeout(id);
  return response;
};

class MCPInspector {
  constructor(baseUrl = '/mcp') {
    this.baseUrl = baseUrl;
    this.userId = 'test_user'; // Replace with actual user auth
    this.accessToken = localStorage.getItem('access_token');
  }

  async sendRequest(method, params = {}) {
    try {
      const response = await fetchWithTimeout(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.accessToken}`,
        },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: `mcp.${method}`,
          params: { user_id: this.userId, ...params },
          id: Date.now(),
        }),
      });
      const data = await response.json();
      if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);
      return data.result;
    } catch (err) {
      console.error(`MCP ${method} error:`, err);
      throw err;
    }
  }

  async initialize() {
    return this.sendRequest('initialize');
  }

  async listTools() {
    return this.sendRequest('listTools');
  }

  async callTool(toolName, args = {}) {
    return this.sendRequest('callTool', { tool_name: toolName, args });
  }

  async listResources() {
    return this.sendRequest('listResources');
  }

  async readResource(uri) {
    return this.sendRequest('readResource', { uri });
  }

  async listPrompts() {
    return this.sendRequest('listPrompts');
  }

  async getPrompt(promptId) {
    return this.sendRequest('getPrompt', { prompt_id: promptId });
  }

  async ping() {
    return this.sendRequest('ping');
  }

  async createMessage(message) {
    return this.sendRequest('createMessage', { message });
  }

  async setLevel(level) {
    return this.sendRequest('setLevel', { level });
  }

  async getSystemMetrics() {
    return this.sendRequest('getSystemMetrics');
  }
}

export default MCPInspector;
