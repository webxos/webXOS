class MCPClient {
  constructor(baseUrl = 'https://webxos-mcp-gateway.onrender.com/v1') {
    this.baseUrl = baseUrl;
    this.token = localStorage.getItem('access_token') || null;
  }

  async request(method, params, id = null) {
    const payload = {
      jsonrpc: "2.0",
      method,
      params,
      id
    };
    try {
      const response = await fetch(`${this.baseUrl}/mcp`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.token}`
        },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`MCP request failed for ${method}: ${error.message}`);
      return { jsonrpc: "2.0", error: { code: -32603, message: error.message }, id };
    }
  }

  async initialize(params) {
    return await this.request("initialize", params);
  }

  async initialized() {
    return await this.request("initialized", {});
  }

  async listTools() {
    return await this.request("tools/list", {});
  }

  async callTool(name, args) {
    return await this.request("tools/call", { name, arguments: args });
  }

  async listResources() {
    return await this.request("resources/list", {});
  }

  async readResource(uri) {
    return await this.request("resources/read", { uri });
  }

  async listPrompts() {
    return await this.request("prompts/list", {});
  }

  async getPrompt(name, args) {
    return await this.request("prompts/get", { name, arguments: args });
  }
}

export { MCPClient };
