exports.handler = async (event) => {
  const { path } = event;
  console.error(`404 Error: Requested path ${path} not found at ${new Date().toISOString()}`);
  return {
    statusCode: 404,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      error: {
        code: -32004,
        message: `Endpoint ${path} not found`,
        timestamp: new Date().toISOString()
      }
    })
  };
};
