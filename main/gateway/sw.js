const CACHE_NAME = 'webxos-ide-v1';
const urlsToCache = [
    '/gateway/server.html',
    'https://cdn.jsdelivr.net/npm/ace-builds@1.4.12/src-min-noconflict/ace.js',
    '/gateway/icon.png'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', event => {
    const requestUrl = new URL(event.request.url);
    if (requestUrl.pathname.includes('/run') || requestUrl.pathname.includes('/save') || requestUrl.pathname.includes('/game')) {
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then(cache => {
                        cache.put(event.request, responseClone);
                    });
                    return response;
                })
                .catch(() => caches.match(event.request)
                    .then(response => response || new Response(JSON.stringify({ error: 'Offline' }), {
                        status: 503,
                        headers: { 'Content-Type': 'application/json' }
                    })))
        );
    } else {
        event.respondWith(
            caches.match(event.request)
                .then(response => response || fetch(event.request))
        );
    }
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.filter(name => name !== CACHE_NAME)
                    .map(name => caches.delete(name))
            );
        })
    );
});
