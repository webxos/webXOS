/**
 * Jina AI Agent for frontend resource retrieval.
 */
async function jinaAgentGetResources(walletId, accessToken, limit = 10) {
    /**
     * Retrieve the latest resources for a wallet.
     * @param {string} walletId - Wallet ID for verification.
     * @param {string} accessToken - OAuth access token.
     * @param {number} limit - Maximum number of resources to retrieve (default: 10).
     * @returns {Promise<Object>} - List of resources.
     */
    try {
        const response = await redaxios.post("https://localhost:8000/api/resources/latest", 
            { wallet_id: walletId, limit }, 
            { headers: { "Authorization": `Bearer ${accessToken}` } }
        );
        await logToConsole(`Retrieved ${response.data.resources.length} resources for wallet ${walletId}`);
        return response.data;
    } catch (error) {
        await logToConsole(`Resource retrieval failed: ${error.message}`);
        throw new Error(`Resource retrieval failed: ${error.message}`);
    }
}