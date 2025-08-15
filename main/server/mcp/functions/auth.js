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
  } catch (e) {
    return {
      statusCode: 400,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32700, message: 'Invalid JSON input' } })
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
      return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          access_token: 'mock_token_456',
          vials: ['vial1', 'vial2'],
          expires_in: 3600
        })
      };
    }
    throw new Error('Invalid credentials');
  } catch (error) {
    return {
      statusCode: 401,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32603, message: error.message } })
    };
  }
};
