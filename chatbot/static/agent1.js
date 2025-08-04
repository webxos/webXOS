async function agent1Search(query, siteIndex) {
    if (!siteIndex || !Array.isArray(siteIndex)) return [];
    const fuseOptions = {
        includeScore: true,
        includeMatches: true,
        threshold: 0.3, // Agent1: More precise search
        keys: ['text.keywords']
    };
    const fuse = new Fuse(siteIndex, fuseOptions);
    return fuse.search(query);
}
