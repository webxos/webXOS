exports.handler = async (event, context) => {
    console.log('Ping function invoked:', event);
    return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'ok', timestamp: new Date().toISOString() })
    };
};
