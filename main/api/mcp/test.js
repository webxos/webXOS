const { NeonDB } = require('./lib/database');
const { signToken } = require('./lib/auth_manager');
const logger = require('./lib/logger');
const fetch = require('node-fetch');

async function runTests() {
  console.log('Running Vial MCP backend tests...');
  const db = new NeonDB();
  const baseUrl = process.env.NODE_ENV === 'production' ? '/.netlify/functions' : 'http://localhost:8888/.netlify/functions';
  const token = await signToken({ user_id: 'test_user_123' });

  // Test 1: Health Endpoint
  console.log('Test 1: Health Endpoint');
  try {
    const healthResponse = await fetch(`${baseUrl}/health`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    const healthData = await healthResponse.json();
    console.assert(healthResponse.status === 200, `Expected 200, got ${healthResponse.status}`);
    console.assert(healthData.status === 'healthy', `Expected healthy, got ${healthData.status}`);
  } catch (error) {
    logger.error(`Health test failed: ${error.message}`);
  }

  // Test 2: Auth Endpoint
  console.log('Test 2: Auth Endpoint');
  try {
    const authResponse = await fetch(`${baseUrl}/auth`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ api_key: 'api-bd9d62ec-a074-4548-8c83-fb054715a870' })
    });
    const authData = await authResponse.json();
    console.assert(authResponse.status === 200, `Expected 200, got ${authResponse.status}`);
    console.assert(authData.access_token, `Expected access_token, got ${authData.access_token}`);
  } catch (error) {
    logger.error(`Auth test failed: ${error.message}`);
  }

  // Test 3: Wallet Update
  console.log('Test 3: Wallet Update');
  try {
    const walletResponse = await fetch(`${baseUrl}/wallet-update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ to_address: 'mock_address_456', amount: 10 })
    });
    const walletData = await walletResponse.json();
    console.assert(walletResponse.status === 200 || walletResponse.status === 401, `Expected 200 or 401, got ${walletResponse.status}`);
  } catch (error) {
    logger.error(`Wallet update test failed: ${error.message}`);
  }

  console.log('Backend tests completed');
}

runTests().catch(error => logger.error(`Test suite failed: ${error.message}`));
