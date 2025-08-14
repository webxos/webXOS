/**
 * Nomic Agent for frontend authentication with OAuth (public copy).
 */
async function nomicAgentAuthenticate(apiKey, walletId) {
    /**
     * Authenticate user with API key and wallet ID.
     * @param {string} apiKey - API key for authentication.
     * @param {string} walletId - Wallet ID for verification.
     * @returns {Promise<Object>} - Authentication result with tokens.
     */
    try {
        const response = await axios.post("https://localhost:8000/api/auth/login", { api_key: apiKey, wallet_id: walletId }, {
            headers: { "X-API-Key": apiKey }
        });
        await logToConsole(`Authenticated wallet ${walletId}`);
        return response.data;
    } catch (error) {
        await logToConsole(`Authentication failed: ${error.message}`);
        throw new Error(`Authentication failed: ${error.message}`);
    }
}

async function nomicAgentRefreshToken(refreshToken, walletId) {
    /**
     * Refresh OAuth token for a wallet.
     * @param {string} refreshToken - Refresh token.
     * @param {string} walletId - Wallet ID for verification.
     * @returns {Promise<Object>} - New access token and expiry.
     */
    try {
        const response = await axios.post("https://localhost:8000/api/auth/refresh", { refresh_token: refreshToken, wallet_id: walletId });
        await logToConsole(`Token refreshed for wallet ${walletId}`);
        return response.data;
    } catch (error) {
        await logToConsole(`Token refresh failed: ${error.message}`);
        throw new Error(`Token refresh failed: ${error.message}`);
    }
}
