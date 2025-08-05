const CACHE_NAME = 'webxos-server-v2';
const urlsToCache = [
    '/server.html',
    '/static/style.css',
    '/static/icon.png',
    '/static/neurots.js',
    '/static/manifest.json',
    '/static/neural_network.wasm',
    '/static/fuse.min.js',
    '/mcp/mcp.js',
    '/utils/search.js',
    '/utils/site_index.json'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});

self.addEventListener('activate', event => {
    const cacheWhitelist = [CACHE_NAME];
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (!cacheWhitelist.includes(cacheName)) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});
