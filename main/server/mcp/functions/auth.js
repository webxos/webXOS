// main/server/mcp/functions/auth.js
async function login(username, password) {
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  if (!response.ok) throw new Error(`Login failed: ${await response.text()}`);
  const { token, userId } = await response.json();
  localStorage.setItem('apiKey', token);
  localStorage.setItem('userId', userId);
  return { token, userId };
}

async function logout() {
  const token = localStorage.getItem('apiKey');
  if (!token) return;
  try {
    await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/auth/logout`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
  } finally {
    localStorage.removeItem('apiKey');
    localStorage.removeItem('userId');
  }
}

async function startWebAuthnRegistration(userId) {
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/auth/webauthn/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId })
  });
  if (!response.ok) throw new Error(`WebAuthn registration failed: ${await response.text()}`);
  const options = await response.json();
  const credential = await navigator.credentials.create({ publicKey: options });
  return credential;
}

async function verifyWebAuthnRegistration(userId, credential) {
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/auth/webauthn/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, credential })
  });
  if (!response.ok) throw new Error(`WebAuthn verification failed: ${await response.text()}`);
  const { token } = await response.json();
  localStorage.setItem('apiKey', token);
  localStorage.setItem('userId', userId);
  return token;
}

export { login, logout, startWebAuthnRegistration, verifyWebAuthnRegistration };
