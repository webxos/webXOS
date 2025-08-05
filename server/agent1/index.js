const { handleHealthCheck } = require('../utils/agent-base');

exports.handler = async (event) => {
    if (event.path.endsWith('/health')) {
        return handleHealthCheck('server-agent1');
    }
    return {
        statusCode: 404,
        body: JSON.stringify({ error: 'Endpoint not found' })
    };
};
