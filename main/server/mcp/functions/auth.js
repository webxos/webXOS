// main/server/mcp/functions/auth.js
import { callTool } from './mcp.js';

export async function createSession(userId) {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('apiKey')}`,
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.createSession',
        params: { user_id: userId, mfa_verified: false },
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    localStorage.setItem('access_token', data.result.access_token);
    return data.result;
  } catch (error) {
    throw new Error(`Failed to create session: ${error.message}`);
  }
}

export async function initiateMFA(userId, mfaMethod) {
  try {
    const response = await callTool('initiate_mfa', { user_id: userId, mfa_method: mfaMethod });
    return response;
  } catch (error) {
    throw new Error(`Failed to initiate MFA: ${error.message}`);
  }
}

export async function verifyMFA(userId, mfaToken, mfaCode) {
  try {
    const response = await callTool('verify_mfa', { user_id: userId, mfa_token: mfaToken, mfa_code: mfaCode });
    return response;
  } catch (error) {
    throw new Error(`Failed to verify MFA: ${error.message}`);
  }
}

export async function revokeSession(sessionId, userId) {
  try {
    const response = await callTool('revoke_session', { session_id: sessionId, user_id: userId });
    return response;
  } catch (error) {
    throw new Error(`Failed to revoke session: ${error.message}`);
  }
}
