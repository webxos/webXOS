exports.handler = async (event, context) => {
  return {
    statusCode: 404,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ error: 'Endpoint not found', path: event.path })
  };
};
