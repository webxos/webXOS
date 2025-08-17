import { AuthTokenOutput } from '../../src/types/api.ts';

const CLIENT_ID = 'YOUR_GITHUB_CLIENT_ID'; // Set in environment or build config
const REDIRECT_URI = 'https://webxos.netlify.app/auth/callback';
const AUTH_URL = `https://github.com/login/oauth/authorize?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&scope=user`;

function generateCodeVerifier(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return btoa(String.fromCharCode(...array)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function generateCodeChallenge(verifier: string): Promise<string> {
  return crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
    .then(buffer => {
      return btoa(String.fromCharCode(...new Uint8Array(buffer)))
        .replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
    });
}

async function initiateOAuthFlow(): Promise<void> {
  const codeVerifier = generateCodeVerifier();
  localStorage.setItem('code_verifier', codeVerifier);
  const codeChallenge = await generateCodeChallenge(codeVerifier);
  
  window.location.href = `${AUTH_URL}&code_challenge=${codeChallenge}&code_challenge_method=S256`;
}

async function handleCallback(): Promise<void> {
  const urlParams = new URLSearchParams(window.location.search);
  const code = urlParams.get('code');
  
  if (code) {
    const codeVerifier = localStorage.getItem('code_verifier');
    if (!codeVerifier) {
      document.getElementById('output').innerText = 'Error: No code verifier found';
      return;
    }
    
    try {
      const response = await fetch('https://webxos.netlify.app/mcp/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'auth.exchangeToken',
          params: { code, redirect_uri: REDIRECT_URI, code_verifier: codeVerifier },
          id: Math.floor(Math.random() * 1000)
        })
      });
      
      const data = await response.json();
      if (data.error) {
        document.getElementById('output').innerText = `Authentication error: ${data.error.message}`;
        return;
      }
      
      const result: AuthTokenOutput = data.result;
      localStorage.setItem('access_token', result.access_token);
      localStorage.setItem('refresh_token', result.refresh_token);
      document.cookie = `session_id=${result.session_id}; HttpOnly; Secure; SameSite=Strict; Max-Age=900`;
      document.getElementById('user-id').innerText = result.user_id;
      document.getElementById('output').innerText = 'Authentication successful';
      
      localStorage.removeItem('code_verifier');
      window.history.replaceState({}, document.title, '/');
      
      const event = new CustomEvent('auth-success', { detail: { user_id: result.user_id } });
      document.dispatchEvent(event);
    } catch (error) {
      document.getElementById('output').innerText = `Authentication error: ${error.message}`;
    }
  }
}

document.getElementById('auth-btn').addEventListener('click', initiateOAuthFlow);

if (window.location.pathname === '/auth/callback') {
  handleCallback();
}
