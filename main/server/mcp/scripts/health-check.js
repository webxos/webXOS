const fetch = require('node-fetch');

async function checkEndpoints() {
  const baseUrl = 'https://webxos.netlify.app/api';
  const endpoints = ['troubleshoot', 'auth/oauth', 'health'];

  for (const endpoint of endpoints) {
    try {
      const url = `${baseUrl}/${endpoint}`;
      const response = await fetch(url, {
        method: endpoint === 'health' ? 'GET' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: endpoint === 'auth/oauth' ? JSON.stringify({ provider: 'mock', code: 'test_code' }) : null
      });
      if (!response.ok) {
        throw new Error(`❌ Error checking ${endpoint}: HTTP ${response.status} - ${await response.text()}`);
      }
      console.log(`✅ ${endpoint} endpoint healthy`);
    } catch (error) {
      console.error(`❌ Error checking ${endpoint}:`, error.message);
      process.exit(1);
    }
  }
  
  console.log('✅ All endpoints healthy');
}

checkEndpoints();
