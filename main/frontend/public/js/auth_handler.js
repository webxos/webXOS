const GOOGLE_CLIENT_ID = 'YOUR_GOOGLE_CLIENT_ID'; // Set in environment or replace
const API_URL = 'http://localhost:8000/mcp';

async function initGoogleSignIn() {
  await new Promise((resolve) => {
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.onload = resolve;
    document.head.appendChild(script);
  });

  google.accounts.id.initialize({
    client_id: GOOGLE_CLIENT_ID,
    callback: handleGoogleResponse,
    state: generateCsrfToken()
  });

  google.accounts.id.renderButton(
    document.getElementById('google-login-btn'),
    { theme: 'outline', size: 'large' }
  );
}

function generateCsrfToken() {
  return Math.random().toString(36).substring(2);
}

async function handleGoogleResponse(response) {
  try {
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'authentication',
        params: { oauth_token: response.credential, provider: 'google' },
        id: 1
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    updateUI(data.result);
    // Dispatch auth-success event for WebSocket connection
    const event = new CustomEvent('auth-success', { detail: { user_id: data.result.user_id } });
    document.dispatchEvent(event);
  } catch (error) {
    document.getElementById('output').innerText = `Error: ${error.message}`;
  }
}

async function fetchUserData(token) {
  try {
    const res = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'vial-management.getUserData',
        params: {},
        id: 2
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    document.getElementById('output').innerText = `Error: ${error.message}`;
    throw error;
  }
}

function updateUI(authData) {
  document.getElementById('user-id').innerText = authData.user_id;
  document.getElementById('status').innerText = 'Authenticated';
  document.getElementById('mode').innerText = 'Online';
  fetchUserData(authData.access_token).then(userData => {
    document.getElementById('balance').innerText = `${userData.balance} $WEBXOS`;
    document.getElementById('reputation').innerText = userData.reputation;
    document.getElementById('wallet-address').innerText = userData.wallet_address;
  });
}

document.getElementById('auth-btn').addEventListener('click', () => {
  google.accounts.id.prompt();
});

initGoogleSignIn();
