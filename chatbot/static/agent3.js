async function agent2Search(query, index) {
    try {
        if (typeof Fuse === 'undefined') {
            throw new Error('Fuse.js not loaded');
        }
        const fuse = new Fuse(index || [], {
            keys: ['text.content'],
            threshold: 0.4,
            includeMatches: true
        });
        const results = fuse.search(query);
        console.log('Agent2 results:', results);
        return results;
    } catch (error) {
        console.error('Agent2 search failed:', error);
        return [];
    }
}

export { agent2Search };
