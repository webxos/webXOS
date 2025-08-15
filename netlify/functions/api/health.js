exports.handler = async (event, context) => {
  try {
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: 'System healthy', uptime: process.uptime(), timestamp: new Date().toISOString() })
    };
  } catch (error) {
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Health check failed', details: error.message })
    };
  }
};
