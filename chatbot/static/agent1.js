async function agent1Search(query, index) {
    try {
        if (typeof Fuse === 'undefined') {
            throw new Error('Fuse.js not loaded');
        }
        const fuse = new Fuse(index || [], {
            keys: ['text.content'],
            threshold: 0.3,
            includeMatches: true
        });
        const results = fuse.search(query);
        console.log('Agent1 results:', results);
        return results;
    } catch (error) {
        console.error('Agent1 search failed:', error);
        return [];
    }
}

export { agent1Search };
