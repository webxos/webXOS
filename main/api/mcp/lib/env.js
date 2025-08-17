const logger = require('./logger');

function detectEnvironment() {
  const env = process.env.NODE_ENV || 'development';
  logger.info(`Detected environment: ${env}`);
  return env;
}

function getConfig() {
  const env = detectEnvironment();
  const config = {
    apiBaseUrl: env === 'production' ? '/.netlify/functions' : 'http://localhost:8888/.netlify/functions',
    dbUrl: process.env.NEON_DATABASE_URL,
    jwtSecret: process.env.JWT_SECRET,
    isDemo: env === 'demo',
    logLevel: env === 'production' ? 'info' : 'debug'
  };

  if (!config.dbUrl || !config.jwtSecret) {
    logger.error('Missing required environment variables: NEON_DATABASE_URL or JWT_SECRET');
    throw new Error('Environment configuration incomplete');
  }

  return config;
}

module.exports = { detectEnvironment, getConfig };
