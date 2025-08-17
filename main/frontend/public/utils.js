export function formatBalance(balance) {
  return balance.toFixed(4);
}

export function formatTimestamp(timestamp) {
  return new Date(timestamp).toISOString().replace('T', ' ').substring(0, 19);
}

export function validateCommand(command) {
  const validCommands = ['/auth', '/void', '/troubleshoot', '/quantum_link', '/export', '/import', '/api_access', '/help'];
  return validCommands.includes(command.trim());
}

export function mockVialData(vialId) {
  return {
    id: vialId,
    status: 'Stopped',
    balance: 0,
    wallet: { address: `mock_${vialId}_address`, balance: 0 }
  };
}

export function handleError(error, logCallback) {
  logCallback(`Error: ${error.message}`);
  if (error.message.includes('Invalid')) {
    localStorage.removeItem('access_token');
    window.location.reload();
  }
}
