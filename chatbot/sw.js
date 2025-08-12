const CACHE_NAME = 'webxos-searchbot-v25';
const urlsToCache = [
    '/chatbot/chatbot.html',
    '/chatbot/static/style.css',
    '/site_index.json',
    '/chatbot/static/icon.png'
];

// Install event: Cache essential files
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Service Worker: Caching files');
                return cache.addAll(urlsToCache);
            })
            .catch(err => console.error('Service Worker: Cache failed', err))
    );
    self.skipWaiting(); // Activate immediately
});

// Fetch event: Serve from cache or network
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    console.log('Service Worker: Serving from cache', event.request.url);
                    return response;
                }
                console.log('Service Worker: Fetching from network', event.request.url);
                return fetch(event.request).catch(() => {
                    console.error('Service Worker: Network fetch failed', event.request.url);
                    return new Response('Offline: Resource unavailable', { status: 503 });
                });
            })
    );
});

// Activate event: Clean up old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames
                    .filter(name => name !== CACHE_NAME)
                    .map(name => {
                        console.log('Service Worker: Deleting old cache', name);
                        return caches.delete(name);
                    })
            );
        })
    );
    self.clients.claim(); // Take control immediately
});
