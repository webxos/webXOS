const { handleHealthCheck } = require('../utils/agent-base');

exports.handler = async (event, context) => {
  if (event.path === '/api/server-agent4/health') {
    return handleHealthCheck('server-agent4');
  }
  return {
    statusCode: 404,
    body: JSON.stringify({ error: 'Endpoint not found' })
  };
};
