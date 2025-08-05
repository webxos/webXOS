const CACHE_NAME = 'webxos-server-v3'; // Updated cache version to force refresh
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
            .then(cache => {
                console.log('Service Worker: Caching files');
                return cache.addAll(urlsToCache);
            })
            .catch(err => console.error('Service Worker: Cache failed', err))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    return response;
                }
                return fetch(event.request).catch(err => {
                    console.error('Service Worker: Fetch failed', err);
                    return new Response('Offline: Resource not available', { status: 503 });
                });
            })
    );
});

self.addEventListener('activate', event => {
    const cacheWhitelist = [CACHE_NAME];
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (!cacheWhitelist.includes(cacheName)) {
                        console.log('Service Worker: Deleting old cache', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});
