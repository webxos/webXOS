// main/server/mcp/functions/mcp.js
export async function listResources() {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.listResources',
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    throw new Error(`Failed to list resources: ${error.message}`);
  }
}

export async function callTool(toolName, params) {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('apiKey')}`,
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.callTool',
        params: { tool_name: toolName, ...params },
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    throw new Error(`Failed to call tool ${toolName}: ${error.message}`);
  }
}
