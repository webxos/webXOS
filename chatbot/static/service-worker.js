const CACHE_NAME = 'webxos-cache';
const urlsToCache = [
  '/chatbot/static/chatbot.html',
  '/chatbot/static/style.css',
  '/chatbot/static/site_index.json',
  '/chatbot/static/agent1.js',
  '/chatbot/static/agent2.js',
  '/chatbot/static/agent3.js',
  '/chatbot/static/agent4.js',
  '/chatbot/static/agentic.js',
  '/chatbot/static/neurots.js',
  '/chatbot/static/sync.js',
  '/chatbot/static/nlp.js',
  '/chatbot/static/learn.js',
  '/chatbot/static/tree.txt',
  'https://cdn.jsdelivr.net/npm/fuse.js@6/dist/fuse.min.js',
  'https://cdn.jsdelivr.net/npm/gun@0.2020/gun.min.js',
  'https://cdn.jsdelivr.net/npm/compromise@14/builds/compromise.min.js',
  'https://cdn.jsdelivr.net/npm/brain.js@2.0.0-beta.24/dist/brain-browser.min.js'
];
const FALLBACK_INDEX = { site_index: [] };

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache).catch(error => {
        console.error('Cache addAll failed:', error);
      });
    })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(cachedResponse => {
      if (cachedResponse) return cachedResponse;
      return fetch(event.request).then(networkResponse => {
        if (!networkResponse.ok && event.request.url.includes('site_index.json')) {
          console.warn('Fetch failed for site_index.json, using fallback');
          return new Response(JSON.stringify(FALLBACK_INDEX), {
            headers: { 'Content-Type': 'application/json' }
          });
        }
        return caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });
      }).catch(error => {
        if (event.request.url.includes('site_index.json')) {
          console.warn('Network fetch failed for site_index.json, using fallback');
          return new Response(JSON.stringify(FALLBACK_INDEX), {
            headers: { 'Content-Type': 'application/json' }
          });
        }
        throw error;
      });
    })
  );
});
