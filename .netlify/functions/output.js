exports.handler = async (event) => {
    const agent = event.queryStringParameters.agent;
    const sessionData = JSON.parse(event.body || '{}');
    const agentData = sessionData[agent] || [];
    const output = agentData.find(item => item.task === 'output')?.output || { error: `No output found for ${agent}` };

    return {
        statusCode: 200,
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify(output)
    };
};
