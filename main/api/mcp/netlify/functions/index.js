const { builder } = require('@netlify/functions');
const authHandler = require('./auth');
const walletHandler = require('./wallet');
const blockchainHandler = require('./blockchain');
const healthHandler = require('./health');

exports.handler = builder(async (event, context) => {
  const path = event.path.replace('/.netlify/functions', '');
  switch (path) {
    case '/auth':
      return authHandler(event, context);
    case '/wallet':
      return walletHandler(event, context);
    case '/blockchain':
      return blockchainHandler(event, context);
    case '/health':
      return healthHandler(event, context);
    default:
      return {
        statusCode: 404,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        },
        body: JSON.stringify({ error: 'Endpoint not found' })
      };
  }
});
