const CACHE_NAME = 'vial-mcp-cache-v1';
const urlsToCache = [
  '/index.html',
  '/styles.css',
  '/utils.js',
  '/config.js',
  '/vial-manager.js',
  '/error-handler.js',
  '/transaction-manager.js',
  '/command-processor.js',
  '/ui-updater.js',
  '/offline-storage.js',
  '/manifest.json'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache);
    })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      if (response) {
        return response;
      }
      return fetch(event.request).catch(() => {
        return caches.match('/index.html');
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
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
