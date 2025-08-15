// main/server/mcp/functions/resources.js
import { callTool } from './mcp.js';

export async function getSystemMetrics() {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('apiKey')}`,
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.getSystemMetrics',
        params: {},
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    throw new Error(`Failed to get system metrics: ${error.message}`);
  }
}

export async function getResourceUsage(vialId) {
  try {
    const response = await callTool('get_resource_usage', { vial_id: vialId });
    return response;
  } catch (error) {
    throw new Error(`Failed to get resource usage: ${error.message}`);
  }
}

export async function allocateResources(vialId, resources) {
  try {
    const response = await callTool('allocate_resources', { vial_id: vialId, resources });
    return response;
  } catch (error) {
    throw new Error(`Failed to allocate resources: ${error.message}`);
  }
}
