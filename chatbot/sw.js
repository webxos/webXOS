self.addEventListener('install', event => {
    event.waitUntil(
        caches.open('webxos-chatbot').then(cache =>
            cache.addAll([
                '/chatbot/chatbot.html',
                '/chatbot/styles.css',
                '/chatbot/script.js',
                '/chatbot/searchData.json',
                '/chatbot/manifest.json'
            ])
        )
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(response => response || fetch(event.request))
    );
});
