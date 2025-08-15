const { writeFileSync } = require('fs');

exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32601, message: 'Method not allowed' } })
    };
  }

  let data;
  try {
    data = event.body ? JSON.parse(event.body) : {};
    if (typeof data !== 'object' || data === null) throw new Error('Invalid JSON object');
  } catch (e) {
    writeFileSync('/tmp/auth_error.log', `Parse Error: ${e.message} at ${new Date().toISOString()}\n`);
    return {
      statusCode: 400,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32700, message: `Invalid JSON: ${e.message}` } })
    };
  }

  const { provider, code } = data;
  if (!provider || !code) {
    return {
      statusCode: 400,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32602, message: 'Missing provider or code' } })
    };
  }

  try {
    if (provider === 'mock' && code === 'test_code') {
      const response = {
        access_token: 'mock_token_xyz',
        vials: ['vial1', 'vial2'],
        expires_in: 3600,
        timestamp: new Date().toISOString()
      };
      writeFileSync('/tmp/auth_success.log', `Success: ${JSON.stringify(response)} at ${new Date().toISOString()}\n`);
      return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(response)
      };
    }
    throw new Error('Invalid credentials');
  } catch (error) {
    writeFileSync('/tmp/auth_error.log', `Auth Error: ${error.message} at ${new Date().toISOString()}\n`, { flag: 'a' });
    if (error.message.includes('not found')) {
      return {
        statusCode: 404,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ error: { code: -32604, message: 'Endpoint not found' } })
      };
    }
    return {
      statusCode: 401,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32603, message: error.message } })
    };
  }
};
