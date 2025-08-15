import CryptoJS from 'crypto-js';

const clientId = 'your_github_client_id'; // Replace with your GitHub OAuth App client ID
const redirectUri = 'https://your-site.netlify.app'; // Replace with your Netlify URL

const generateCodeVerifier = () => {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
};

const generateCodeChallenge = (verifier) => {
  const hashed = CryptoJS.SHA256(verifier);
  return hashed.toString(CryptoJS.enc.Base64).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};

export const initiateOAuth = async () => {
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = generateCodeChallenge(codeVerifier);
  sessionStorage.setItem('code_verifier', codeVerifier);

  const authUrl = `https://github.com/login/oauth/authorize?client_id=${clientId}&redirect_uri=${redirectUri}&scope=read:user&response_type=code&code_challenge=${codeChallenge}&code_challenge_method=S256`;

  return authUrl;
};

export const handleCallback = async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const code = urlParams.get('code');
  const codeVerifier = sessionStorage.getItem('code_verifier');

  if (!code || !codeVerifier) {
    throw new Error('Missing OAuth code or verifier');
  }

  const response = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      client_id: clientId,
      code,
      code_verifier: codeVerifier,
      redirect_uri: redirectUri,
    }),
  });

  const data = await response.json();
  if (data.error) {
    throw new Error(data.error_description || 'OAuth token exchange failed');
  }

  localStorage.setItem('oauth_token', data.access_token);
  sessionStorage.removeItem('code_verifier');
};
