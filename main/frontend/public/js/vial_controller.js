import { JSONRPCRequest, JSONRPCResponse, WalletBalanceOutput, VialGitPushOutput, AuthTokenOutput } from '../../src/types/api.ts';

const API_URL = 'https://webxos.netlify.app/mcp/execute';

async function executeAPI<T>(request: JSONRPCRequest): Promise<T> {
  const accessToken = localStorage.getItem('access_token');
  const sessionId = document.cookie.match(/session_id=([^;]+)/)?.[1];
  if (!accessToken || !sessionId) {
    document.getElementById('output').innerText = 'Session expired. Please re-authenticate.';
    triggerReauthentication();
    throw new Error('Not authenticated');
  }
  
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${accessToken}`,
      'X-Session-ID': sessionId
    },
    body: JSON.stringify(request)
  });
  
  const data: JSONRPCResponse = await response.json();
  if (data.error) {
    if (data.error.message.includes("Invalid token or session")) {
      document.getElementById('output').innerText = 'Session expired. Please re-authenticate.';
      triggerReauthentication();
    }
    throw new Error(data.error.message);
  }
  return data.result as T;
}

function triggerReauthentication() {
  localStorage.removeItem('access_token');
  document.cookie = 'session_id=; Max-Age=0; path=/;';
  document.getElementById('user-id').innerText = 'Not logged in';
  document.getElementById('execute-claude-btn').disabled = true;
  document.getElementById('quantum-link-btn').disabled = true;
  document.getElementById('mine-btn').disabled = true;
  document.getElementById('export-btn').disabled = true;
  document.getElementById('api-credentials-btn').disabled = true;
  document.getElementById('git-push-btn').disabled = true;
  document.getElementById('cash-out-btn').disabled = true;
  document.getElementById('logout-btn').disabled = true;
  document.getElementById('auth-btn').click(); // Trigger OAuth flow
}

async function updateVialBalances(userId: string) {
  const vialIds = ['vial1', 'vial2', 'vial3', 'vial4'];
  for (const vialId of vialIds) {
    try {
      const result: WalletBalanceOutput = await executeAPI({
        jsonrpc: '2.0',
        method: 'wallet.getVialBalance',
        params: { user_id: userId, vial_id: vialId },
        id: Math.floor(Math.random() * 1000)
      });
      document.getElementById(`${vialId}-status`).innerText = `Running (Balance: ${result.balance.toFixed(4)})`;
    } catch (error) {
      document.getElementById('output').innerText = `Error fetching balance for ${vialId}: ${error.message}`;
    }
  }
}

async function handleGitPush(userId: string, vialId: string, code: string, commitMessage: string) {
  try {
    const result: VialGitPushOutput = await executeAPI({
      jsonrpc: '2.0',
      method: 'vial_management.gitPush',
      params: { user_id: userId, vial_id: vialId, commit_message: commitMessage },
      id: Math.floor(Math.random() * 1000)
    });
    document.getElementById('output').innerText = `Git push successful: Commit ${result.commit_hash}, New Balance: ${result.balance}`;
    document.getElementById('balance').innerText = `${result.balance} $WEBXOS`;
  } catch (error) {
    document.getElementById('output').innerText = `Error pushing code: ${error.message}`;
  }
}

async function handleCashOut(userId: string, amount: number, destinationAddress: string) {
  try {
    if (!/^\d+(\.\d{1,4})?$/.test(amount.toString())) {
      throw new Error('Invalid amount: Must be a positive number with up to 4 decimal places');
    }
    if (!/^[a-f0-9]{64}$/.test(destinationAddress)) {
      throw new Error('Invalid destination address: Must be a 64-character hexadecimal string');
    }
    const result = await executeAPI({
      jsonrpc: '2.0',
      method: 'wallet.cashOut',
      params: { user_id: userId, amount, destination_address: destinationAddress },
      id: Math.floor(Math.random() * 1000)
    });
    document.getElementById('output').innerText = `Cash-out successful: Transaction ${result.transaction_id}, New Balance: ${result.new_balance}`;
    document.getElementById('balance').innerText = `${result.new_balance} $WEBXOS`;
    await updateTransactionHistory(userId);
  } catch (error) {
    document.getElementById('output').innerText = `Cash-out error: ${error.message}`;
  }
}

async function updateTransactionHistory(userId: string) {
  try {
    const result = await executeAPI({
      jsonrpc: '2.0',
      method: 'wallet.getTransactions',
      params: { user_id: userId },
      id: Math.floor(Math.random() * 1000)
    });
    const transactions = result.transactions || [];
    const historyDiv = document.getElementById('transaction-history');
    historyDiv.innerHTML = transactions.map(tx => `
      <div class="p-2 border-b">
        <p><strong>Transaction ID:</strong> ${tx.transaction_id}</p>
        <p><strong>Amount:</strong> ${tx.amount} $WEBXOS</p>
        <p><strong>Destination:</strong> ${tx.destination_address}</p>
        <p><strong>Timestamp:</strong> ${new Date(tx.timestamp).toLocaleString()}</p>
      </div>
    `).join('');
  } catch (error) {
    document.getElementById('output').innerText = `Error fetching transaction history: ${error.message}`;
  }
}

document.addEventListener('auth-success', async (event: CustomEvent) => {
  const userId: string = event.detail.user_id;
  await updateVialBalances(userId);
  await updateTransactionHistory(userId);
});

document.getElementById('git-push-btn').addEventListener('click', async () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  const code = document.getElementById('agent-code').value;
  const commitMessage = document.getElementById('commit-message').value || 'Update agent code';
  if (!code) {
    document.getElementById('output').innerText = 'Error: No agent code provided';
    return;
  }
  await handleGitPush(userId, 'vial1', code, commitMessage);
});

document.getElementById('cash-out-btn').addEventListener('click', async () => {
  const userId = document.getElementById('user-id').innerText;
  if (userId === 'Not logged in') {
    document.getElementById('output').innerText = 'Error: Please authenticate first';
    return;
  }
  const amount = parseFloat(document.getElementById('cash-out-amount').value);
  const destinationAddress = document.getElementById('cash-out-address').value;
  if (!amount || !destinationAddress) {
    document.getElementById('output').innerText = 'Error: Please provide amount and destination address';
    return;
  }
  await handleCashOut(userId, amount, destinationAddress);
});
