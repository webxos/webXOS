const { healthCheck, logError } = require('../utils/agent-base');

exports.handler = async (event, context) => {
    const path = event.path.split('/').pop();
    if (path === 'health') {
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(await healthCheck())
        };
    } else if (path === 'log') {
        const { message, stack } = JSON.parse(event.body);
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(await logError(message, stack, 'agent1'))
        };
    }
    return {
        statusCode: 404,
        body: JSON.stringify({ error: 'Not found' })
    };
};
