const CACHE_NAME = 'vial-mcp-v2';
const urlsToCache = [
  '/',
  '/index.html',
  '/css/tailwind.min.css',
  '/js/auth_handler.js',
  '/js/websocket_handler.js',
  '/js/vial_controller.js',
  '/manifest.json'
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

self.addEventListener('message', event => {
  if (event.data.action === 'cachePendingImport') {
    caches.open(CACHE_NAME).then(cache => {
      cache.put('/pending-import', new Response(JSON.stringify(event.data.data)));
    });
  }
});

self.addEventListener('sync', event => {
  if (event.tag === 'sync-pending-import') {
    event.waitUntil(
      caches.open(CACHE_NAME).then(cache =>
        cache.match('/pending-import').then(response =>
          response ? response.json().then(data => {
            fetch('http://localhost:8000/mcp/execute', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                jsonrpc: '2.0',
                method: 'wallet.importWallet',
                params: data,
                id: Math.floor(Math.random() * 1000)
              })
            }).then(res => res.json()).then(result => {
              if (!result.error) {
                cache.delete('/pending-import');
              }
            });
          }) : Promise.resolve()
        )
      )
    );
  }
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
