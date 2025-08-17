import { JSONRPCRequest, JSONRPCResponse, AuthTokenOutput } from '../../src/types/api.ts';

const GITHUB_CLIENT_ID = 'YOUR_GITHUB_CLIENT_ID'; // Replace with actual Client ID from Netlify env
const REDIRECT_URI = 'https://webxos.netlify.app/auth/callback';
const API_URL = 'https://webxos.netlify.app/mcp/execute';

function generateCodeVerifier() {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return btoa(String.fromCharCode.apply(null, array))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

async function generateCodeChallenge(verifier) {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return btoa(String.fromCharCode(...new Uint8Array(digest)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

async function initiateOAuth() {
  const codeVerifier = generateCodeVerifier();
  localStorage.setItem('code_verifier', codeVerifier);
  const codeChallenge = await generateCodeChallenge(codeVerifier);
  const state = crypto.randomUUID();
  localStorage.setItem('oauth_state', state);
  
  const authUrl = `https://github.com/login/oauth/authorize?client_id=${GITHUB_CLIENT_ID}&redirect_uri=${encodeURIComponent(REDIRECT_URI)}&scope=user&code_challenge=${codeChallenge}&code_challenge_method=S256&state=${state}`;
  window.location.href = authUrl;
}

async function handleOAuthCallback() {
  const urlParams = new URLSearchParams(window.location.search);
  const code = urlParams.get('code');
  const state = urlParams.get('state');
  const storedState = localStorage.getItem('oauth_state');
  
  if (!code || state !== storedState) {
    document.getElementById('output').innerText = 'OAuth error: Invalid state or code';
    return;
  }
  
  try {
    const codeVerifier = localStorage.getItem('code_verifier');
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'auth.exchangeToken',
        params: { code, redirect_uri: REDIRECT_URI, code_verifier: codeVerifier },
        id: Math.floor(Math.random() * 1000)
      })
    });
    const data: JSONRPCResponse = await response.json();
    if (data.error) throw new Error(data.error.message);
    
    const { access_token, user_id, session_id } = data.result as AuthTokenOutput;
    localStorage.setItem('access_token', access_token);
    document.cookie = `session_id=${session_id}; HttpOnly; Secure; SameSite=Strict; Max-Age=900`;
    document.getElementById('user-id').innerText = user_id;
    document.getElementById('output').innerText = 'Authentication successful!';
    
    // Trigger WebSocket connection
    const event = new CustomEvent('auth-success', { detail: { user_id } });
    document.dispatchEvent(event);
    
    // Clean up
    localStorage.removeItem('code_verifier');
    localStorage.removeItem('oauth_state');
    window.history.replaceState({}, document.title, '/');
  } catch (error) {
    document.getElementById('output').innerText = `Authentication error: ${error.message}`;
  }
}

document.getElementById('google-login-btn').style.display = 'none'; // Hide Google login
document.getElementById('auth-btn').addEventListener('click', initiateOAuth);

// Check for OAuth callback on page load
if (window.location.pathname === '/auth/callback') {
  handleOAuthCallback();
}
