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
    event.waitUntil(syncWithRetry());
  }
});

async function syncWithRetry(maxRetries = 3, retryDelay = 5000) {
  let attempts = 0;
  while (attempts < maxRetries) {
    try {
      const cache = await caches.open(CACHE_NAME);
      const response = await cache.match('/pending-import');
      if (!response) return;
      
      const data = await response.json();
      const res = await fetch('http://localhost:8000/mcp/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'wallet.importWallet',
          params: data,
          id: Math.floor(Math.random() * 1000)
        })
      });
      const result = await res.json();
      if (!result.error) {
        await cache.delete('/pending-import');
        self.clients.matchAll().then(clients => {
          clients.forEach(client => client.postMessage({ action: 'sync-complete', data }));
        });
        return;
      }
      throw new Error(result.error.message);
    } catch (error) {
      attempts++;
      if (attempts < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, retryDelay));
        continue;
      }
      self.clients.matchAll().then(clients => {
        clients.forEach(client => client.postMessage({ action: 'sync-failed', error: error.message }));
      });
    }
  }
}

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
