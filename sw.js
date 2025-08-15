self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('vial-mcp-cache').then((cache) => {
      return cache.addAll([
        '/index.html',
      ]);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          caches.open('vial-mcp-cache').then((cache) => {
            cache.put(event.request, networkResponse.clone());
          });
        }
        return networkResponse;
      }).catch(() => {
        return caches.match('/index.html');
      });
    })
  );
});
