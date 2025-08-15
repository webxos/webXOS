exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32601, message: 'Method not allowed' } })
    };
  }

  const { provider, code } = JSON.parse(event.body || '{}');
  if (!provider || !code) {
    return {
      statusCode: 400,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32602, message: 'Missing provider or code' } })
    };
  }

  try {
    if (provider === 'mock' && code === 'test_code') {
      return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          access_token: 'mock_token_123',
          vials: ['vial1', 'vial2']
        })
      };
    }
    throw new Error('Invalid credentials');
  } catch (error) {
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32603, message: error.message } })
    };
  }
};
