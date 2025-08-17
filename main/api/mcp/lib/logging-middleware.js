const logger = require('./logger');

function loggingMiddleware(event, context, next) {
  const start = Date.now();
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  
  logger.info(`Request started: ${requestId}, method: ${event.httpMethod}, path: ${event.path}`);

  return next(event, context).then(response => {
    const duration = Date.now() - start;
    logger.info(`Request completed: ${requestId}, status: ${response.statusCode}, duration: ${duration}ms`);
    return response;
  }).catch(error => {
    logger.error(`Request failed: ${requestId}, error: ${error.message}`);
    throw error;
  });
}

module.exports = { loggingMiddleware };
