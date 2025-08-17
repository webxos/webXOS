import { config } from './config.js';

export function generateMockData() {
  return {
    wallet: {
      balance: config.DEFAULT_BALANCE,
      address: config.DEFAULT_WALLET_ADDRESS,
      user_id: config.DEFAULT_USER_ID,
      reputation: config.DEFAULT_REPUTATION
    },
    vials: config.VIALS.map(id => ({
      id,
      status: 'Stopped',
      balance: 0,
      wallet: { address: `mock_${id}_address`, balance: 0 }
    })),
    transactions: [],
    credentials: {
      api_key: 'WEBXOS-MOCKKEY',
      api_secret: 'MOCKSECRET1234567890'
    }
  };
}

export function mockApiResponse(endpoint) {
  const mockData = generateMockData();
  const mocks = {
    '/health': {
      status: 'healthy',
      balance: mockData.wallet.balance,
      reputation: mockData.wallet.reputation,
      user_id: mockData.wallet.user_id,
      address: mockData.wallet.address,
      vials: mockData.vials
    },
    '/auth': {
      access_token: 'mock_token',
      token_type: 'bearer',
      expires_in: 86400
    },
    '/credentials': mockData.credentials,
    '/blockchain': { transactions: mockData.transactions },
    '/import-export': { export_data: generateExportData(mockData) },
    '/quantum-link': { message: 'Quantum link activated', block: { id: 'mock_block', hash: 'mock_hash' }, reward: 10 },
    '/void': { message: 'System voided successfully' },
    '/troubleshoot': { status: 'healthy', vials_count: 4, transactions_count: 0, blocks_count: 0 }
  };
  return mocks[endpoint] || { error: `No mock for ${endpoint}` };
}

function generateExportData(mockData) {
  return `# Vial MCP Export\n\n## Wallet\n- Balance: ${mockData.wallet.balance.toFixed(4)} $WEBXOS\n- Address: ${mockData.wallet.address}\n- User ID: ${mockData.wallet.user_id}\n- Reputation: ${mockData.wallet.reputation}\n\n## Vials\n${mockData.vials.map(vial => `# Vial ${vial.id}\n- Status: ${vial.status}\n- Balance: ${vial.balance.toFixed(4)} $WEBXOS\n- Wallet Address: ${vial.wallet.address}\n`).join('---\n')}`;
}
