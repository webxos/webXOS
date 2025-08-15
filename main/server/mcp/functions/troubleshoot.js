exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32601, message: 'Method not allowed' } })
    };
  }

  try {
    // Simulate system check
    const status = { status: 'OK', details: 'System check completed at 05:15 AM EDT' };
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(status)
    };
  } catch (error) {
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32603, message: error.message } })
    };
  }
};
