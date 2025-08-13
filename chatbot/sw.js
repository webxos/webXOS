self.addEventListener('install', event => {
    event.waitUntil(
        caches.open('vial-mcp-cache').then(cache => {
            return cache.addAll([
                '/chatbot.html',
                '/chatbot/chatbot2.html',
                '/static/style.css',
                '/vial/static/redaxios.min.js',
                '/vial/static/dexie.min.js',
                '/chatbot/site_index.json'
            ]);
        })
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(response => {
            return response || fetch(event.request).then(fetchResponse => {
                return caches.open('vial-mcp-cache').then(cache => {
                    cache.put(event.request, fetchResponse.clone());
                    return fetchResponse;
                });
            });
        }).catch(error => {
            console.error('Service Worker error:', error);
            fetch('/api/audit_log', {
                method: 'POST',
                body: JSON.stringify({ user_id: 'service_worker', action: `error: ${error.message}`, wallet: { webxos: 0.0, transactions: [] } })
            });
        })
    );
});
