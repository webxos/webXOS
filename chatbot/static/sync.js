const gun = Gun(['https://gun-manhattan.herokuapp.com/gun']);

function initGun() {
    try {
        gun.get('webxos-search').put({ initialized: true, timestamp: new Date().toISOString() });
        console.log('Gun.js initialized');
    } catch (error) {
        console.error('Gun.js initialization failed:', error);
    }
}

function syncResults(agent, query, results) {
    try {
        const timestamp = new Date().toISOString();
        const data = {
            agent,
            query,
            results: results.map(r => ({
                content: r.item.text.content,
                path: r.item.path || '',
                source: r.item.source || 'WebXOS'
            })),
            timestamp
        };
        gun.get('webxos-search').get(timestamp).put(data);
        console.log(`Synced results for ${agent}: ${query}`);
    } catch (error) {
        console.error('Sync failed:', error);
    }
}

export { initGun, syncResults };
