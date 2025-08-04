const CACHE_NAME = 'webxos-searchbot-v18';
const urlsToCache = [
    '/chatbot/static/chatbot.html',
    '/chatbot/static/style.css',
    '/chatbot/static/site_index.json',
    '/chatbot/static/agent1.js',
    '/chatbot/static/agent2.js',
    '/chatbot/static/agent3.js',
    '/chatbot/static/agent4.js',
    '/chatbot/static/agentic.js',
    '/chatbot/static/neurots.js',
    '/chatbot/static/sync.js',
    '/chatbot/static/nlp.js',
    '/chatbot/static/tree.txt',
    '/chatbot/static/fuse.min.js',
    'https://cdn.jsdelivr.net/npm/gun@0.2020/gun.min.js',
    'https://cdn.jsdelivr.net/npm/compromise@14/builds/compromise.min.js'
];
const FALLBACK_INDEX = { site_index: [] };

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(urlsToCache).catch(error => {
                console.error('Cache addAll failed:', error);
            });
        })
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(cachedResponse => {
            if (cachedResponse) return cachedResponse;
            return fetch(event.request).then(networkResponse => {
                if (!networkResponse.ok && (event.request.url.includes('site_index.json') || event.request.url.endsWith('./site_index.json'))) {
                    console.warn(`Fetch failed for ${event.request.url} (Base: ${self.location.origin}), using fallback`);
                    return new Response(JSON.stringify(FALLBACK_INDEX), {
                        headers: { 'Content-Type': 'application/json' }
                    });
                }
                if (!networkResponse.ok && event.request.url.includes('sync.js')) {
                    console.error(`Failed to fetch sync.js: HTTP ${networkResponse.status} ${networkResponse.statusText}`);
                }
                return caches.open(CACHE_NAME).then(cache => {
                    cache.put(event.request, networkResponse.clone());
                    return networkResponse;
                });
            }).catch(error => {
                if (event.request.url.includes('site_index.json') || event.request.url.endsWith('./site_index.json')) {
                    console.warn(`Network fetch failed for ${event.request.url} (Base: ${self.location.origin}), using fallback`);
                    return new Response(JSON.stringify(FALLBACK_INDEX), {
                        headers: { 'Content-Type': 'application/json' }
                    });
                }
                console.error(`Fetch error for ${event.request.url}:`, error);
                throw error;
            });
        })
    );
});
