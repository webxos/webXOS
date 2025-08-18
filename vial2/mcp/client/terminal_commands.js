import { mcpStatus, mcpConnect, mcpListResources, mcpGetResource, mcpListPrompts, mcpGetPrompt } from './mcp_client.js';

export async function handleTerminalCommand(cmd, log) {
    const parts = cmd.trim().split(' ').filter(p => p);
    if (!localStorage.getItem('token') && cmd !== '/auth' && cmd !== '/help') {
        log('Please authenticate first using /auth');
        return;
    }
    if (cmd === '/help') {
        log(`
            Available commands:
            /auth - Initiate OAuth2.0 authentication
            /mcp status - Check MCP server status
            /mcp connect <vial_id> [server] [port] - Connect to MCP server
            /mcp tools/list - List available MCP tools
            /mcp tools/call <tool> [key=value ...] - Call an MCP tool
            /mcp resources/list - List available resources
            /mcp resources/get <resource> - Get a resource
            /mcp prompts/list - List available prompts
            /mcp prompts/get <prompt> - Get a prompt
            /git <command> - Run git command (status, pull, push, commit -m)
            /git model <action> - Git model operations (commit_model, push_model, pull_model)
            /training <action> - Training operations (start_training, commit_training)
            /api_key generate <vial_id> - Generate API key for vial
            /quantum link <vial_id> - Link vial to quantum network
            /wallet sync <vial_id> - Sync wallet with WebXOS
            /mine - Start proof-of-work mining
            /api - Request API access
        `);
    } else if (cmd === '/auth') {
        log('Redirecting to OAuth2.0 authentication...');
        window.location.href = '/mcp/auth';
    } else if (cmd.startsWith('/mcp')) {
        const subCommand = parts[1];
        try {
            if (subCommand === 'status') {
                const status = await mcpStatus();
                log(JSON.stringify(status, null, 2));
            } else if (subCommand === 'connect') {
                const vialId = parts[2];
                const server = parts[3] || 'default';
                const port = parts[4] || 6277;
                const result = await mcpConnect(vialId, server, port);
                log(JSON.stringify(result, null, 2));
            } else if (subCommand === 'resources/list') {
                const resources = await mcpListResources();
                log(JSON.stringify(resources, null, 2));
            } else if (subCommand === 'resources/get') {
                const resourceUri = parts[2];
                const resource = await mcpGetResource(resourceUri);
                log(JSON.stringify(resource, null, 2));
            } else if (subCommand === 'prompts/list') {
                const prompts = await mcpListPrompts();
                log(JSON.stringify(prompts, null, 2));
            } else if (subCommand === 'prompts/get') {
                const promptName = parts[2];
                const prompt = await mcpGetPrompt(promptName);
                log(JSON.stringify(prompt, null, 2));
            } else {
                log('Unknown MCP command');
            }
        } catch (e) {
            log(`Error: ${e.message}`);
        }
    } else if (cmd.startsWith('/git model')) {
        try {
            const action = parts[2];
            const response = await axios.post('/mcp/api/vial/git/model', { vial_id: 'vial1', action }, {
                headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
            });
            log(JSON.stringify(response.data.result.data, null, 2));
        } catch (e) {
            log(`Error: ${e.response?.data?.error?.message || e.message}`);
        }
    } else if (cmd.startsWith('/training')) {
        try {
            const action = parts[1];
            const response = await axios.post('/mcp/api/vial/training/pipeline', { vial_id: 'vial1', action }, {
                headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
            });
            log(JSON.stringify(response.data.result.data, null, 2));
        } catch (e) {
            log(`Error: ${e.response?.data?.error?.message || e.message}`);
        }
    } else if (cmd.startsWith('/api_key generate')) {
        try {
            const vialId = parts[2];
            const response = await axios.post('/mcp/api/vial/api_key/generate', { vial_id: vialId }, {
                headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
            });
            log(JSON.stringify(response.data.result.data, null, 2));
        } catch (e) {
            log(`Error: ${e.response?.data?.error?.message || e.message}`);
        }
    } else if (cmd.startsWith('/quantum link')) {
        try {
            const vialId = parts[2];
            const response = await axios.post('/mcp/api/vial/quantum/link', { vial_id: vialId }, {
                headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
            });
            log(JSON.stringify(response.data.result.data, null, 2));
        } catch (e) {
            log(`Error: ${e.response?.data?.error?.message || e.message}`);
        }
    } else if (cmd.startsWith('/wallet sync')) {
        try {
            const vialId = parts[2];
            const response = await axios.post('/mcp/api/vial/wallet/sync', { vial_id: vialId }, {
                headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
            });
            log(JSON.stringify(response.data.result.data, null, 2));
        } catch (e) {
            log(`Error: ${e.response?.data?.error?.message || e.message}`);
        }
    } else {
        log('Unknown command. Type /help for commands.');
    }
}

// xAI Artifact Tags: #vial2 #mcp #client #javascript #inspector #git #neon_mcp
