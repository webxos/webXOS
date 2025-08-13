self.addEventListener('install', event => {
  event.waitUntil(
    caches.open('vial-mcp-cache').then(cache => {
      return cache.addAll([
        '/static/style.css',
        '/vial/static/dexie.min.js',
        '/vial/static/redaxios.min.js'
      ]);
    })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
