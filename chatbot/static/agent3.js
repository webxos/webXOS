

async function agent3Search(query, siteIndex) {
    if (!siteIndex || !Array.isArray(siteIndex)) return [];
    const fuseOptions = {
        includeScore: true,
        includeMatches: true,
        threshold: 0.5, // Agent3: Broader search
        keys: ['text.keywords']
    };
    const fuse = new Fuse(siteIndex, fuseOptions);
    return fuse.search(query);
}
