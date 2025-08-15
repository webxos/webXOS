// main/server/mcp/functions/resources.js
async function getSystemMetrics() {
  const token = localStorage.getItem('apiKey');
  const userId = localStorage.getItem('userId');
  if (!token || !userId) throw new Error('Not authenticated');
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/resources/metrics`, {
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (!response.ok) throw new Error(`Failed to fetch system metrics: ${await response.text()}`);
  return await response.json();
}

async function getResourceById(resourceId) {
  const token = localStorage.getItem('apiKey');
  const userId = localStorage.getItem('userId');
  if (!token || !userId) throw new Error('Not authenticated');
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/resources/${resourceId}`, {
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (!response.ok) throw new Error(`Failed to fetch resource: ${await response.text()}`);
  return await response.json();
}

async function listResources(category = null) {
  const token = localStorage.getItem('apiKey');
  const userId = localStorage.getItem('userId');
  if (!token || !userId) throw new Error('Not authenticated');
  const url = category
    ? `${process.env.API_BASE || 'http://localhost:8000'}/resources?category=${encodeURIComponent(category)}`
    : `${process.env.API_BASE || 'http://localhost:8000'}/resources`;
  const response = await fetch(url, {
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (!response.ok) throw new Error(`Failed to list resources: ${await response.text()}`);
  return await response.json();
}

export { getSystemMetrics, getResourceById, listResources };
