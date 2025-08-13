self.addEventListener('install', event => {
  event.waitUntil(
    caches.open('llmware-cache').then(cache => {
      return cache.addAll([
        '/static/style.css',
        '/vial/static/dexie.min.js',
        '/vial/static/redaxios.min.js'
      ]);
    })
  );
});

self.addEventListener('fetch', event => {
  if (event.request.url.includes('/api/library/3')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        const fetched = fetch(event.request).then(response => {
          const clone = response.clone();
          caches.open('llmware-cache').then(cache => cache.put(event.request, clone));
          return response.json().then(data => {
            return new Response(JSON.stringify({
              data: data.response,
              wallet: data.wallet,
              formatted: `LLMware result: ${JSON.stringify(data.response.result)}`
            }));
          });
        });
        return cached || fetched;
      }).catch(error => {
        return new Response(JSON.stringify({ error: `LLMware fetch error: ${error.message}` }), { status: 500 });
      })
    );
  } else {
    event.respondWith(fetch(event.request));
  }
});
