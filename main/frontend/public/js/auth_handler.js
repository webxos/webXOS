const API_URL = 'http://localhost:8000/mcp';
const GOOGLE_CLIENT_ID = 'your_google_client_id'; // Replace with actual client ID from .env

// Load Google Sign-In SDK
function loadGoogleSignIn() {
  const script = document.createElement('script');
  script.src = 'https://accounts.google.com/gsi/client';
  script.async = true;
  script.defer = true;
  document.head.appendChild(script);
}

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
    document.getElementById('wallet-address').innerText = `wallet_${userId}`;
  } catch (error) {
    document.getElementById('output').innerText = `Authentication error: ${error.message}`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadGoogleSignIn();
  
  // Initialize Google Sign-In
  window.google.accounts.id.initialize({
    client_id: GOOGLE_CLIENT_ID,
    callback: (response) => {
      authenticateWithGoogle(response.credential);
    }
  });
  
  // Render Google Sign-In button
  window.google.accounts.id.renderButton(
    document.getElementById('google-login-btn'),
    { theme: 'outline', size: 'large' }
  );
});

document.getElementById('auth-btn').addEventListener('click', () => {
  document.getElementById('output').innerText = 'Please use Google Login for authentication';
});
