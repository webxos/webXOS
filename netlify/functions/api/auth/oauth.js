exports.handler = async (event, context) => {
  try {
    const redirectUrl = 'https://mock-oauth-provider.com/login?client_id=abc123&redirect_uri=https://webxos.netlify.app/api/auth/callback';
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ redirect: redirectUrl })
    };
  } catch (error) {
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'OAuth failed', details: error.message })
    };
  }
};
