// main/server/mcp/functions/auth.js
export async function login(username, password) {
  try {
    const response = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    localStorage.setItem('apiKey', data.access_token);
    localStorage.setItem('userId', data.user_id);
    return data;
  } catch (error) {
    throw new Error(`Login failed: ${error.message}`);
  }
}

export async function loginWithWallet(walletAddress) {
  try {
    const response = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wallet_address: walletAddress }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    localStorage.setItem('apiKey', data.access_token);
    localStorage.setItem('userId', data.user_id);
    return data;
  } catch (error) {
    throw new Error(`Wallet login failed: ${error.message}`);
  }
}

export function logout() {
  localStorage.removeItem('apiKey');
  localStorage.removeItem('userId');
}
