exports.handler = async (event) => {
  if (event.httpMethod !== 'GET') {
    return {
      statusCode: 405,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32601, message: 'Method not allowed' } })
    };
  }

  try {
    const response = {
      status: "healthy",
      time: "07:36 AM EDT",
      timestamp: new Date().toISOString()
    };
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(response)
    };
  } catch (error) {
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32603, message: error.message } })
    };
  }
};
