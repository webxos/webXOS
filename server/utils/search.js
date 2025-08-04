const Fuse = require('fuse.js');
const fetch = require('node-fetch');

async function loadSiteIndex(fs, path, indexPath) {
    try {
        const data = await fs.readFile(path.join(__dirname, indexPath), 'utf8');
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

async function performSearch(query, siteIndex, agentId, threshold = 0.3, useLLM = false) {
    if (useLLM) {
        try {
            const llmResponse = await fetch('https://x.ai/api/grok', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.LLM_API_KEY}`
                },
                body: JSON.stringify({ query, agent: agentId })
            });
            const llmResults = await llmResponse.json();
            return llmResults; // Process LLM results
        } catch (error) {
            console.error(`${agentId} LLM search failed:`, error.message);
            return [];
        }
    }
    if (!siteIndex || !Array.isArray(siteIndex)) return [];
    const fuseOptions = {
        includeScore: true,
        includeMatches: true,
        threshold,
        keys: ['text.keywords']
    };
    const fuse = new Fuse(siteIndex, fuseOptions);
    return fuse.search(query);
}

module.exports = { loadSiteIndex, performSearch };
