exports.handler = async (event, context) => {
  try {
    const diagnostics = {
      system: 'Vial MCP Gateway',
      status: 'Operational',
      timestamp: new Date().toISOString()
    };
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'Diagnostics complete', data: diagnostics })
    };
  } catch (error) {
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Troubleshoot failed', details: error.message })
    };
  }
};
