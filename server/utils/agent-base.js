exports.handleHealthCheck = (agentName) => {
  return {
    statusCode: 200,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      status: 'Healthy',
      agent: agentName,
      timestamp: new Date().toISOString()
    })
  };
};
