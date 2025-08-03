async function agent4Search(query, siteIndex) {
    if (!siteIndex || !Array.isArray(siteIndex)) return [];
    const fuseOptions = {
        includeScore: true,
        includeMatches: true,
        threshold: 0.6, // Agent4: Very broad search
        keys: ['text.keywords']
    };
    const fuse = new Fuse(siteIndex, fuseOptions);
    return fuse.search(query);
}
