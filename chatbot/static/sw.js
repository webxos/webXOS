const CACHE_NAME = 'webxos-searchbot-v25';
const urlsToCache = [
    '/chatbot/chatbot.html',
    '/chatbot/static/style.css',
    '/chatbot/static/icon.png',
    '/site_index.json',
    '/manifest.json'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(urlsToCache);
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.filter(name => name !== CACHE_NAME).map(name => caches.delete(name))
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(response => {
            return response || fetch(event.request).then(networkResponse => {
                if (networkResponse.ok && event.request.method === 'GET') {
                    const clone = networkResponse.clone();
                    caches.open(CACHE_NAME).then(cache => {
                        cache.put(event.request, clone);
                    });
                }
                return networkResponse;
            });
        }).catch(() => {
            return new Response('Offline: Resource unavailable', { status: 503 });
        })
    );
});
