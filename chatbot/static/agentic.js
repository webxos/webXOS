import { syncResults } from './sync.js';
import { processQuery } from './nlp.js';

async function agenticSearch(query, siteIndex) {
    const processedQuery = processQuery(query);
    if (!processedQuery.topics.some(t => ['webxos', 'decentralized', 'pwa', 'webgl'].includes(t.toLowerCase()))) {
        return [{ item: { text: { content: 'Agentic: Sorry, I’m designed to assist with WebXOS-related queries. Try asking about our decentralized features!' } } }];
    }
    const agentPromises = [
        window.agent1Search(query, siteIndex),
        window.agent2Search(query, siteIndex),
        window.agent3Search(query, siteIndex),
        window.agent4Search(query, siteIndex)
    ];
    const results = (await Promise.all(agentPromises)).flat().map(result => ({
        item: { text: { content: `Agentic: ${result.item.text.content}` }, path: result.item.path, source: result.item.source }
    }));
    await syncResults('agentic', query, results);
    return results;
}

export { agenticSearch };
