async function agent4Search(query, index) {
    try {
        if (typeof Fuse === 'undefined') {
            throw new Error('Fuse.js not loaded');
        }
        const fuse = new Fuse(index || [], {
            keys: ['text.content'],
            threshold: 0.6,
            includeMatches: true
        });
        const results = fuse.search(query);
        console.log('Agent4 results:', results);
        return results;
    } catch (error) {
        console.error('Agent4 search failed:', error);
        return [];
    }
}

export { agent4Search };
