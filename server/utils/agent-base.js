const handleHealthCheck = (agentName) => {
    return {
        statusCode: 200,
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET'
        },
        body: JSON.stringify({
            status: 'Healthy',
            agent: agentName,
            timestamp: new Date().toISOString(),
            metrics: {
                cpu: Math.random() * 100, // Mock CPU usage
                memory: Math.random() * 100, // Mock memory usage
                uptime: Math.floor(Math.random() * 10000) // Mock uptime in seconds
            }
        })
    };
};

module.exports = { handleHealthCheck };
