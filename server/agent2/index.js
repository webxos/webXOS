const express = require('express');
const Fuse = require('fuse.js');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const port = process.env.PORT || 3002;

app.use(express.json());

async function loadSiteIndex() {
    try {
        const data = await fs.readFile(path.join(__dirname, '../utils/site_index.json'), 'utf8');
        const siteIndex = JSON.parse(data);
        if (!siteIndex || !Array.isArray(siteIndex)) {
            throw new Error('site_index.json is empty or invalid');
        }
        return siteIndex;
    } catch (error) {
        console.error('Failed to load site_index.json:', error.message);
        return [];
    }
}

async function agent2Search(query, siteIndex, useLLM = false) {
    if (useLLM) {
        try {
            const llmResponse = await fetch('https://x.ai/api/grok', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.LLM_API_KEY}`
                },
                body: JSON.stringify({ query })
            });
            const llmResults = await llmResponse.json();
            return llmResults; // Process LLM results
        } catch (error) {
            console.error('LLM search failed:', error.message);
            return [];
        }
    }
    if (!siteIndex || !Array.isArray(siteIndex)) return [];
    const fuseOptions = {
        includeScore: true,
        includeMatches: true,
        threshold: 0.4,
        keys: ['text.keywords']
    };
    const fuse = new Fuse(siteIndex, fuseOptions);
    return fuse.search(query);
}

app.post('/search', async (req, res) => {
    const { query, useLLM = false } = req.body;
    if (!query || typeof query !== 'string') {
        return res.status(400).json({ error: 'Invalid or missing query parameter' });
    }

    try {
        const siteIndex = await loadSiteIndex();
        const results = await agent2Search(query, siteIndex, useLLM);
        res.json({
            success: true,
            results: results.map(({ item, matches, score }) => ({
                item,
                matches,
                score
            }))
        });
    } catch (error) {
        console.error('Search error:', error.message);
        res.status(500).json({ error: 'Search failed', details: error.message });
    }
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', agent: 'Agent2', version: '1.0.0' });
});

app.post('/control', async (req, res) => {
    const { command } = req.body;
    if (!command) {
        return res.status(400).json({ error: 'Missing command' });
    }
    try {
        switch (command) {
            case 'status':
                res.json({ status: 'healthy', agent: 'Agent2', version: '1.0.0' });
                break;
            case 'restart':
                res.json({ message: 'Restart not implemented yet. Placeholder response.' });
                break;
            default:
                res.status(400).json({ error: `Unknown command: ${command}` });
        }
    } catch (error) {
        res.status(500).json({ error: `Control command failed: ${error.message}` });
    }
});

if (process.env.NODE_ENV !== 'production') {
    app.listen(port, () => {
        console.log(`Agent2 server running on port ${port}`);
    });
}

module.exports.handler = async (event, context) => {
    const { httpMethod, body, path } = event;
    if (httpMethod === 'POST' && path === '/search') {
        const { query, useLLM = false } = JSON.parse(body || '{}');
        if (!query || typeof query !== 'string') {
            return {
                statusCode: 400,
                body: JSON.stringify({ error: 'Invalid or missing query parameter' })
            };
        }
        try {
            const siteIndex = await loadSiteIndex();
            const results = await agent2Search(query, siteIndex, useLLM);
            return {
                statusCode: 200,
                body: JSON.stringify({
                    success: true,
                    results: results.map(({ item, matches, score }) => ({
                        item,
                        matches,
                        score
                    }))
                })
            };
        } catch (error) {
            console.error('Search error:', error.message);
            return {
                statusCode: 500,
                body: JSON.stringify({ error: 'Search failed', details: error.message })
            };
        }
    } else if (httpMethod === 'GET' && path === '/health') {
        return {
            statusCode: 200,
            body: JSON.stringify({ status: 'healthy', agent: 'Agent2', version: '1.0.0' })
        };
    } else if (httpMethod === 'POST' && path === '/control') {
        const { command } = JSON.parse(body || '{}');
        if (!command) {
            return {
                statusCode: 400,
                body: JSON.stringify({ error: 'Missing command' })
            };
        }
        try {
            switch (command) {
                case 'status':
                    return {
                        statusCode: 200,
                        body: JSON.stringify({ status: 'healthy', agent: 'Agent2', version: '1.0.0' })
                    };
                case 'restart':
                    return {
                        statusCode: 200,
                        body: JSON.stringify({ message: 'Restart not implemented yet. Placeholder response.' })
                    };
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
    }
    return {
        statusCode: 405,
        body: JSON.stringify({ error: 'Method not allowed' })
    };
};
