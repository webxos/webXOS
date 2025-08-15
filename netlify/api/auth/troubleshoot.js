exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { code: -32601, method: 'Method not allowed' } })
    };
  }

  try {
    const response = {
      status: "OK",
      details: "Troubleshoot check completed at 07:36 AM EDT",
      healthy: true
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
