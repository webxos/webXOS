exports.handler = async (event, context) => {
    console.log('Ping function invoked:', {
        path: event.path,
        method: event.httpMethod,
        headers: event.headers,
        timestamp: new Date().toISOString()
    });
    return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'ok', timestamp: new Date().toISOString() })
    };
};
