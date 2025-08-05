exports.handler = async (event, context) => {
  if (event.path === '/api/mcp/log' && event.httpMethod === 'POST') {
    try {
      const { message, timestamp } = JSON.parse(event.body);
      if (!message || !timestamp) {
        return {
          statusCode: 400,
          body: JSON.stringify({ error: 'Missing message or timestamp' })
        };
      }
      console.log(`[${timestamp}] [MCP] ${message}`);
      return {
        statusCode: 200,
        body: JSON.stringify({ status: 'Log received', message, timestamp })
      };
    } catch (err) {
      console.error(`Log Error: ${err.message}`);
      return {
        statusCode: 500,
        body: JSON.stringify({ error: `Log Error: ${err.message}` })
      };
    }
  }
  return {
    statusCode: 404,
    body: JSON.stringify({ error: 'Endpoint not found' })
  };
};
