const GITHUB_CLIENT_ID = 'YOUR_GITHUB_CLIENT_ID'; // Replace with actual Client ID from Netlify env
const REDIRECT_URI = 'https://<your-netlify-site>.netlify.app/auth/callback';
const API_URL = 'https://<your-netlify-site>.netlify.app/mcp/execute';

async function initiateOAuth() {
  const authUrl = `https://github.com/login/oauth/authorize?client_id=${GITHUB_CLIENT_ID}&redirect_uri=${encodeURIComponent(REDIRECT_URI)}&scope=user`;
  window.location.href = authUrl;
}

async function handleOAuthCallback() {
  const urlParams = new URLSearchParams(window.location.search);
  const code = urlParams.get('code');
  if (code) {
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'auth.exchangeToken',
          params: { code, redirect_uri: REDIRECT_URI },
          id: Math.floor(Math.random() * 1000)
        })
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      
      const { access_token, user_id } = data.result;
      localStorage.setItem('access_token', access_token);
      document.getElementById('user-id').innerText = user_id;
      document.getElementById('output').innerText = 'Authentication successful!';
      
      // Trigger WebSocket connection
      const event = new CustomEvent('auth-success', { detail: { user_id } });
      document.dispatchEvent(event);
      
      // Redirect to main page
      window.history.replaceState({}, document.title, '/');
    } catch (error) {
      document.getElementById('output').innerText = `Authentication error: ${error.message}`;
    }
  }
}

document.getElementById('google-login-btn').style.display = 'none'; // Hide Google login
document.getElementById('auth-btn').addEventListener('click', initiateOAuth);

// Check for OAuth callback on page load
if (window.location.pathname === '/auth/callback') {
  handleOAuthCallback();
}
