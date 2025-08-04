const express = require('express');
const fetch = require('node-fetch');

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

const agents = [
    { id: 'agent1', url: '/api/agent1' },
    { id: 'agent2', url: '/api/agent2' },
    { id: 'agent3', url: '/api/agent3' },
    { id: 'agent4', url: '/api/agent4' }
];

async function checkAgentStatus(agent) {
    try {
        const response = await fetch(`${agent.url}/health`, { timeout: 5000 });
        const data = await response.json();
        return { id: agent.id, status: data.status, version: data.version };
    } catch (error) {
        console.error(`Failed to check ${agent.id} status:`, error.message);
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
                res.json({ message: 'Logs not implemented yet. Placeholder response.' });
                break;
            case 'test':
                res.json({ message: 'Test command executed successfully.' });
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

if (process.env.NODE_ENV !== 'production') {
    app.listen(port, () => {
        console.log(`MCP server running on port ${port}`);
    });
}

module.exports.handler = async (event, context) => {
    const { httpMethod, body, path } = event;
    if (httpMethod === 'GET' && path === '/status') {
        try {
            const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
            const activeAgents = agentStatuses
                .filter(agent => agent.status === 'healthy')
                .map(agent => agent.id);
            return {
                statusCode: 200,
                body: JSON.stringify({
                    status: 'healthy',
                    activeAgents,
                    details: agentStatuses
                })
            };
        } catch (error) {
            console.error('MCP status check failed:', error.message);
            return {
                statusCode: 500,
                body: JSON.stringify({ error: 'MCP status check failed', details: error.message })
            };
        }
    } else if (httpMethod === 'POST' && path === '/control') {
        const { command, apiKey } = JSON.parse(body || '{}');
        if (!command) {
            return {
                statusCode: 400,
                body: JSON.stringify({ error: 'Missing command' })
            };
        }
        try {
            switch (command) {
                case 'diagnostics':
                    const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
                    return {
                        statusCode: 200,
                        body: JSON.stringify({
                            message: 'Diagnostics completed',
                            details: agentStatuses
                        })
                    };
                case 'logs':
                    return {
                        statusCode: 200,
                        body: JSON.stringify({ message: 'Logs not implemented yet. Placeholder response.' })
                    };
                case 'test':
                    return {
                        statusCode: 200,
                        body: JSON.stringify({ message: 'Test command executed successfully.' })
                    };
                case 'llm-test':
                    if (!apiKey) {
                        return {
                            statusCode: 400,
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
                            body: JSON.stringify({ message: 'LLM test successful', details: llmResults })
                        };
                    } catch (error) {
                        return {
                            statusCode: 500,
                            body: JSON.stringify({ error: 'LLM test failed', details: error.message })
                        };
                    }
                default:
                    return {
                        statusCode: 400,
                        body: JSON.stringify({ error: `Unknown command: ${command}` })
                    };
            }
        } catch (error) {
            return {
                statusCode: 500,
                body: JSON.stringify({ error: `Control command failed: ${error.message}` })
            };
        }
    } else if (httpMethod === 'POST' && path === '/debug') {
        const { command } = JSON.parse(body || '{}');
        if (!command) {
            return {
                statusCode: 400,
                body: JSON.stringify({ error: 'Missing debug command' })
            };
        }
        try {
            switch (command) {
                case 'ping':
                    return {
                        statusCode: 200,
                        body: JSON.stringify({ message: 'MCP ping successful' })
                    };
                case 'agent-status':
                    const agentStatuses = await Promise.all(agents.map(checkAgentStatus));
                    return {
                        statusCode: 200,
                        body: JSON.stringify({ message: 'Agent status retrieved', details: agentStatuses })
                    };
                default:
                    return {
                        statusCode: 400,
                        body: JSON.stringify({ error: `Unknown debug command: ${command}` })
                    };
            }
        } catch (error) {
            return {
                statusCode: 500,
                body: JSON.stringify({ error: `Debug command failed: ${error.message}` })
            };
        }
    }
    return {
        statusCode: 405,
        body: JSON.stringify({ error: 'Method not allowed' })
    };
};
