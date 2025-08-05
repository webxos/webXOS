self.addEventListener('install', event => {
    event.waitUntil(
        caches.open('webxos-v1').then(cache => {
            return cache.addAll([
                '/',
                '/server.html',
                '/static/style.css',
                '/static/neurots.js',
                '/static/fuse.min.js',
                '/static/manifest.json',
                '/static/icon.png',
                '/utils/search.js',
                '/utils/site_index.json',
                '/mcp/mcp.js'
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
