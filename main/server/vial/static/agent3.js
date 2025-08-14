/**
 * LLMware Agent for frontend note management.
 */
async function llmwareAgentAddNote(content, walletId, accessToken, resourceId = null) {
    /**
     * Add a note for a wallet.
     * @param {string} content - Note content.
     * @param {string} walletId - Wallet ID for verification.
     * @param {string} accessToken - OAuth access token.
     * @param {string|null} resourceId - Optional resource ID.
     * @returns {Promise<Object>} - Success message with note ID.
     */
    try {
        const response = await redaxios.post("https://localhost:8000/api/notes/add", 
            { wallet_id: walletId, content, resource_id: resourceId }, 
            { headers: { "Authorization": `Bearer ${accessToken}` } }
        );
        await logToConsole(`Note ${response.data.note_id} added for wallet ${walletId}`);
        return response.data;
    } catch (error) {
        await logToConsole(`Note add failed: ${error.message}`);
        throw new Error(`Note add failed: ${error.message}`);
    }
}

async function llmwareAgentReadNote(noteId, walletId, accessToken) {
    /**
     * Read a note by ID for a wallet.
     * @param {number} noteId - ID of the note to read.
     * @param {string} walletId - Wallet ID for verification.
     * @param {string} accessToken - OAuth access token.
     * @returns {Promise<Object>} - Note content and metadata.
     */
    try {
        const response = await redaxios.post("https://localhost:8000/api/notes/read", 
            { note_id: noteId, wallet_id: walletId }, 
            { headers: { "Authorization": `Bearer ${accessToken}` } }
        );
        await logToConsole(`Note ${noteId} read for wallet ${walletId}`);
        return response.data;
    } catch (error) {
        await logToConsole(`Note read failed: ${error.message}`);
        throw new Error(`Note read failed: ${error.message}`);
    }
}