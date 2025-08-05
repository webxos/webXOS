exports.healthCheck = async () => {
    // Simulate agent health check
    const status = Math.random() > 0.1 ? 'Healthy' : 'Error';
    return { status, timestamp: new Date().toISOString() };
};

exports.logError = async (message, stack, agentName) => {
    // Placeholder for logging to errorlog.md
    console.log(`[${agentName}] ${message}\n${stack}`);
    return { status: 'Logged' };
};
