export async function mcpStatus() {
    try {
        const response = await axios.post('/mcp/api/vial/mcp/status', {}, {
            headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
        return response.data.result.data;
    } catch (e) {
        throw new Error(`MCP Status failed: ${e.response?.data?.error?.message || e.message}`);
    }
}

export async function mcpConnect(vialId, server = 'default') {
    try {
        const response = await axios.post('/mcp/api/vial/mcp/connect', { vial_id: vialId, server }, {
            headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
        return response.data.result.data;
    } catch (e) {
        throw new Error(`MCP Connect failed: ${e.response?.data?.error?.message || e.message}`);
    }
}

export async function mcpListResources() {
    try {
        const response = await axios.post('/mcp/api/inspector/resources/list', {}, {
            headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
        return response.data.result.data;
    } catch (e) {
        throw new Error(`MCP Resources list failed: ${e.response?.data?.error?.message || e.message}`);
    }
}

export async function mcpGetResource(resourceUri) {
    try {
        const response = await axios.post('/mcp/api/inspector/resources/get', { resource_uri: resourceUri }, {
            headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
        return response.data.result.data;
    } catch (e) {
        throw new Error(`MCP Resource get failed: ${e.response?.data?.error?.message || e.message}`);
    }
}

export async function mcpListPrompts() {
    try {
        const response = await axios.post('/mcp/api/inspector/prompts/list', {}, {
            headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
        return response.data.result.data;
    } catch (e) {
        throw new Error(`MCP Prompts list failed: ${e.response?.data?.error?.message || e.message}`);
    }
}

export async function mcpGetPrompt(promptName) {
    try {
        const response = await axios.post('/mcp/api/inspector/prompts/get', { prompt_name: promptName }, {
            headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
        return response.data.result.data;
    } catch (e) {
        throw new Error(`MCP Prompt get failed: ${e.response?.data?.error?.message || e.message}`);
    }
}

// xAI Artifact Tags: #vial2 #mcp #client #javascript #inspector #neon_mcp
