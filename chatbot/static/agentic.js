import { fuseSearch } from './searchUtils.js';
import { syncResults } from './sync.js';
import { processQuery } from './nlp.js';
import { trainNetwork, predictRelevance } from './learn.js';

async function agenticSearch(query, siteIndex) {
  const processedQuery = processQuery(query);
  if (!processedQuery.topics.some(t => ['webxos', 'decentralized', 'pwa', 'webgl'].includes(t.toLowerCase()))) {
    return [{ item: { text: { content: 'Sorry, I’m designed to assist with WebXOS-related queries. Try asking about our decentralized features!' } } }];
  }
  const results = await Promise.all([
    window.agent1Search(query, siteIndex),
    window.agent2Search(query, siteIndex),
    window.agent3Search(query, siteIndex),
    window.agent4Search(query, siteIndex)
  ]);
  const combinedResults = results.flat();
  await syncResults('agentic', query, combinedResults);
  const relevance = await predictRelevance(query);
  if (relevance > 0.5) {
    await trainNetwork(query, 1);
  }
  return combinedResults;
}

export { agenticSearch };
