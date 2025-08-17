const API_URL = 'http://localhost:8000/mcp';

async function authenticateWithGoogle(credential) {
  try {
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'authentication',
        params: { oauth_token: credential, provider: 'google' },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    
    const userId = data.result.user_id;
    document.getElementById('user-id').innerText = userId;
    document.getElementById('output').innerText = 'Authentication successful!';
    
    // Dispatch auth-success event to trigger WebSocket connection and wallet sync
    const event = new CustomEvent('auth-success', { detail: { user_id: userId } });
    document.dispatchEvent(event);
    
    // Fetch initial wallet data
    const walletRes = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'wallet.getVialBalance',
        params: { user_id: userId, vial_id: 'vial1' },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const walletData = await walletRes.json();
    if (walletData.error) throw new Error(walletData.error.message);
    
    localStorage.setItem(`wallet_${userId}`, JSON.stringify({ balance: walletData.result.balance }));
    document.getElementById('balance').innerText = `${walletData.result.balance} $WEBXOS`;
    document.getElementById('wallet-address').innerText = `wallet_${userId}`; // Simplified for demo
  } catch (error) {
    document.getElementById('output').innerText = `Authentication error: ${error.message}`;
  }
}

document.getElementById('google-login-btn').addEventListener('click', () => {
  // Placeholder for Google OAuth popup
  // In a real implementation, use Google Sign-In SDK
  const mockCredential = 'mock_google_token';
  authenticateWithGoogle(mockCredential);
});

document.getElementById('auth-btn').addEventListener('click', () => {
  document.getElementById('output').innerText = 'Please use Google Login for authentication';
});
