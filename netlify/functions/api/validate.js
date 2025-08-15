exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: JSON.stringify({ error: 'Method not allowed' }) };
  }
  const data = JSON.parse(event.body || '{}');
  if (!data || typeof data !== 'object') {
    return { statusCode: 400, body: JSON.stringify({ error: 'Invalid input' }) };
  }
  return { statusCode: 200, body: JSON.stringify({ status: 'valid', data }) };
};
