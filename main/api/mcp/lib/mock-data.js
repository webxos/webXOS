const logger = require('./logger');

function generateMockUser(user_id) {
  const mockUser = {
    user_id: user_id || 'mock_user_' + Math.random().toString(36).substring(2, 15),
    wallet_address: 'mock_wallet_' + Math.random().toString(36).substring(2, 15),
    balance: 38940.0000,
    reputation: 1200983581,
    username: 'mock_user',
    api_key: 'WEBXOS-MOCKKEY-' + Math.random().toString(36).substring(2, 10),
    api_secret: 'MOCKSECRET' + Math.random().toString(36).substring(2, 10)
  };
  logger.info(`Generated mock user: ${mockUser.user_id}`);
  return mockUser;
}

function generateMockVial(user_id, vial_id) {
  const mockVial = {
    id: vial_id || 'vial' + Math.floor(Math.random() * 4 + 1),
    user_id,
    status: 'Stopped',
    balance: 0,
    wallet_address: 'mock_vial_wallet_' + Math.random().toString(36).substring(2, 15),
    created_at: new Date()
  };
  logger.info(`Generated mock vial: ${mockVial.id}`);
  return mockVial;
}

function generateMockTransaction(user_id) {
  const mockTransaction = {
    transaction_id: 'tx_' + Date.now() + '_' + Math.random().toString(36).substring(2, 15),
    from_address: 'mock_wallet_' + Math.random().toString(36).substring(2, 15),
    to_address: 'mock_wallet_' + Math.random().toString(36).substring(2, 15),
    amount: Math.random() * 100,
    created_at: new Date()
  };
  logger.info(`Generated mock transaction: ${mockTransaction.transaction_id}`);
  return mockTransaction;
}

module.exports = { generateMockUser, generateMockVial, generateMockTransaction };
