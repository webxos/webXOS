import { fuseSearch } from './searchUtils.js';
import { syncResults } from './sync.js';
import { processQuery } from './nlp.js';
import Q from 'https://cdn.jsdelivr.net/npm/q@1.5.1/dist/q.min.js';

async function agenticSearch(query, siteIndex) {
  const processedQuery = processQuery(query);
  if (!processedQuery.topics.some(t => ['webxos', 'decentralized', 'pwa', 'webgl'].includes(t.toLowerCase()))) {
    return [{ item: { text: { content: 'Agentic: Sorry, I’m designed to assist with WebXOS-related queries. Try asking about our decentralized features!' } } }];
  }
  const agentPromises = [
    Q.fcall(() => window.agent1Search(query, siteIndex)),
    Q.fcall(() => window.agent2Search(query, siteIndex)),
    Q.fcall(() => window.agent3Search(query, siteIndex)),
    Q.fcall(() => window.agent4Search(query, siteIndex))
  ];
  const results = await Q.all(agentPromises).then(results => results.flat().map(result => ({
    item: { text: { content: `Agentic: ${result.item.text.content}` } }
  })));
  await syncResults('agentic', query, results);
  return results;
}

export { agenticSearch };
