self.addEventListener('install', event => {
  event.waitUntil(
    caches.open('nomic-cache').then(cache => {
      return cache.addAll([
        '/static/style.css',
        '/vial/static/dexie.min.js',
        '/vial/static/redaxios.min.js'
      ]);
    })
  );
});

self.addEventListener('fetch', event => {
  if (event.request.url.includes('/api/library/1')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        const fetched = fetch(event.request).then(response => {
          const clone = response.clone();
          caches.open('nomic-cache').then(cache => cache.put(event.request, clone));
          return response.json().then(data => {
            return new Response(JSON.stringify({
              data: data.response,
              wallet: data.wallet,
              formatted: `Nomic embeddings: ${JSON.stringify(data.response.embeddings)}`
            }));
          });
        });
        return cached || fetched;
      }).catch(error => {
        return new Response(JSON.stringify({ error: `Nomic fetch error: ${error.message}` }), { status: 500 });
      })
    );
  } else {
    event.respondWith(fetch(event.request));
  }
});
