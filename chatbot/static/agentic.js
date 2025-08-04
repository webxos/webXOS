async function agenticSearch(query, index) {
    try {
        if (typeof Fuse === 'undefined') {
            throw new Error('Fuse.js not loaded');
        }
        const results = await Promise.all([
            agent1Search(query, index),
            agent2Search(query, index),
            agent3Search(query, index),
            agent4Search(query, index)
        ]);
        const combined = results.flat().sort((a, b) => (b.score || 0) - (a.score || 0)).slice(0, 10);
        console.log('Agentic results:', combined);
        return combined;
    } catch (error) {
        console.error('Agentic search failed:', error);
        return [];
    }
}

export { agenticSearch };
