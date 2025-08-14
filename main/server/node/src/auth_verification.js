/**
 * Authentication verification for Node.js fallback server.
 */
function verifyApiKey(apiKey, walletId) {
    /**
     * Verify API key and wallet ID pair.
     * @param {string} apiKey - API key to verify.
     * @param {string} walletId - Wallet ID to verify.
     * @returns {boolean} - True if valid, false otherwise.
     */
    const validKeys = {
        'api-a24cb96b-96cd-488d-a013-91cb8edbbe68': 'wallet_123'
    };
    return validKeys[apiKey] === walletId;
}
module.exports = { verifyApiKey };
