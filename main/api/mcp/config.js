module.exports = {
  API_BASE_URL: process.env.NODE_ENV === 'production' ? '/.netlify/functions' : 'http://localhost:8888/.netlify/functions',
  DEMO_MODE: process.env.NODE_ENV === 'demo',
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 2000,
  BLOCKCHAIN_DIFFICULTY: 4,
  REWARD_AMOUNT: 10,
  REPUTATION_INCREMENT: 100,
  VIALS: ['vial1', 'vial2', 'vial3', 'vial4'],
  LOG_LEVEL: process.env.NODE_ENV === 'production' ? 'info' : 'debug'
};
