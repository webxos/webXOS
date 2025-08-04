const express = require('express');
const fetch = require('node-fetch');

const app = express();
app.use(express.json());
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
});

const agents = [
    { id: 'chatbot-agent1', url: '/.netlify/functions/chatbot-agent1' },
    { id: 'chatbot-agent2', url: '/.netlify/functions/chatbot-agent2' },
    { id: 'chatbot-agent3', url: '/.netlify/functions/chatbot-agent3' },
    { id: 'chatbot-agent4', url: '/.netlify/functions/chatbot-agent4' },
    { id: 'server-agent1', url: '/.netlify/functions/server-agent1' },
    { id: 'server-agent2', url: '/.netlify/functions/server-agent2' },
    { id: 'server-agent3', url: '/.netlify/functions/server-agent3' },
    { id: 'server-agent4', url: '/.netlify/functions/server-agent4' }
];

async function checkAgentStatus(agent) {
    try {
        const response = await fetch(`${agent.url}/health`, { timeout: 5000 });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        return { id: agent.id, status: data.status, version: data.version };
    } catch (error) {
        console.error(`Failed to check ${agent.id} status: ${error.message}`);
        return { id: agent.id, status: 'error', error: error.message };
    }
}

app.get('/status', async (req, res) => {
    try {
        const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
        const activeAgents = agentStatuses
            .filter(agent => agent.status === 'healthy')
            .map(agent => agent.id);
        res.json({
            status: 'healthy',
            activeAgents,
            details: agentStatuses
        });
    } catch (error) {
        console.error('MCP status check failed:', error.message);
        res.status(500).json({ error: 'MCP status check failed', details: error.message });
    }
});

app.post('/control', async (req, res) => {
    const { command, apiKey } = req.body;
    if (!command) {
        return res.status(400).json({ error: 'Missing command' });
    }
    try {
        switch (command) {
            case 'diagnostics':
                const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
                res.json({
                    message: 'Diagnostics completed',
                    details: agentStatuses
                });
                break;
            case 'logs':
                res.json({ message: 'Logs not implemented yet' });
                break;
            case 'test':
                res.json({ message: 'Test command executed successfully' });
                break;
            case 'llm-test':
                if (!apiKey) {
                    return res.status(400).json({ error: 'Missing LLM API key' });
                }
                try {
                    const llmResponse = await fetch('https://x.ai/api/grok', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${apiKey}`
                        },
                        body: JSON.stringify({ query: 'Test LLM integration' })
                    });
                    const llmResults = await llmResponse.json();
                    res.json({ message: 'LLM test successful', details: llmResults });
                } catch (error) {
                    res.status(500).json({ error: 'LLM test failed', details: error.message });
                }
                break;
            default:
                res.status(400).json({ error: `Unknown command: ${command}` });
        }
    } catch (error) {
        res.status(500).json({ error: `Control command failed: ${error.message}` });
    }
});

app.post('/debug', async (req, res) => {
    const { command } = req.body;
    if (!command) {
        return res.status(400).json({ error: 'Missing debug command' });
    }
    try {
        switch (command) {
            case 'ping':
                res.json({ message: 'MCP ping successful' });
                break;
            case 'agent-status':
                const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
                res.json({ message: 'Agent status retrieved', details: agentStatuses });
                break;
            default:
                res.status(400).json({ error: `Unknown debug command: ${command}` });
        }
    } catch (error) {
        res.status(500).json({ error: `Debug command failed: ${error.message}` });
    }
});

module.exports.handler = async (event, context) => {
    const { httpMethod, path, body } = event;
    if (httpMethod === 'GET' && path === '/status') {
        const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
        const activeAgents = agentStatuses
            .filter(agent => agent.status === 'healthy')
            .map(agent => agent.id);
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
            body: JSON.stringify({
                status: 'healthy',
                activeAgents,
                details: agentStatuses
            })
        };
    } else if (httpMethod === 'POST' && path === '/control') {
        const { command, apiKey } = JSON.parse(body || '{}');
        if (!command) {
            return {
                statusCode: 400,
                headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                body: JSON.stringify({ error: 'Missing command' })
            };
        }
        switch (command) {
            case 'diagnostics':
                const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
                return {
                    statusCode: 200,
                    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    body: JSON.stringify({
                        message: 'Diagnostics completed',
                        details: agentStatuses
                    })
                };
            case 'logs':
                return {
                    statusCode: 200,
                    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    body: JSON.stringify({ message: 'Logs not implemented yet' })
                };
            case 'test':
                return {
                    statusCode: 200,
                    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    body: JSON.stringify({ message: 'Test command executed successfully' })
                };
            case 'llm-test':
                if (!apiKey) {
                    return {
                        statusCode: 400,
                        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                        body: JSON.stringify({ error: 'Missing LLM API key' })
                    };
                }
                try {
                    const llmResponse = await fetch('https://x.ai/api/grok', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${apiKey}`
                        },
                        body: JSON.stringify({ query: 'Test LLM integration' })
                    });
                    const llmResults = await llmResponse.json();
                    return {
                        statusCode: 200,
                        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                        body: JSON.stringify({ message: 'LLM test successful', details: llmResults })
                    };
                } catch (error) {
                    return {
                        statusCode: 500,
                        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                        body: JSON.stringify({ error: 'LLM test failed', details: error.message })
                    };
                }
            default:
                return {
                    statusCode: 400,
                    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    body: JSON.stringify({ error: `Unknown command: ${command}` })
                };
        }
    } else if (httpMethod === 'POST' && path === '/debug') {
        const { command } = JSON.parse(body || '{}');
        if (!command) {
            return {
                statusCode: 400,
                headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                body: JSON.stringify({ error: 'Missing debug command' })
            };
        }
        switch (command) {
            case 'ping':
                return {
                    statusCode: 200,
                    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    body: JSON.stringify({ message: 'MCP ping successful' })
                };
            case 'agent-status':
                const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
                return {
                    statusCode: 200,
                    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    body: JSON.stringify({ message: 'Agent status retrieved', details: agentStatuses })
                };
            default:
                return {
                    statusCode: 400,
                    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    body: JSON.stringify({ error: `Unknown debug command: ${command}` })
                };
        }
    }
    return {
        statusCode: 405,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({ error: 'Method not allowed' })
    };
};
