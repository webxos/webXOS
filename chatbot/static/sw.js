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
  'https://cdn.jsdelivr.net/npm/fuse.js@6/dist/fuse.min.js',
  'https://cdn.jsdelivr.net/npm/gun@0.2020/gun.min.js',
  'https://cdn.jsdelivr.net/npm/compromise@14/builds/compromise.min.js',
  'https://cdn.jsdelivr.net/npm/brain.js@2/dist/brain-browser.min.js'
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
      return response || fetch(event.request);
    })
  );
});
