self.addEventListener("install",event=>{
    event.waitUntil(
        caches.open("vial-mcp-cache-v1").then(cache=>
            cache.addAll([
                "/static/style.css",
                "/static/dexie.min.js",
                "/static/redaxios.min.js",
                "/static/agent1.js",
                "/static/agent2.js",
                "/static/agent3.js",
                "/static/agent4.js"
            ])
        ).catch(err=>console.error(`Cache open failed: ${err}`))
    );
    console.log("Service worker installed");
});

self.addEventListener("fetch",event=>{
    if(event.request.method!=="GET")return;
    event.respondWith(
        caches.match(event.request).then(cached=>cached||fetch(event.request).then(response=>{
            const responseClone=response.clone();
            caches.open("vial-mcp-cache-v1").then(cache=>cache.put(event.request,responseClone));
            return response;
        }).catch(err=>{
            console.error(`Fetch failed: ${err}`);
            return new Response(JSON.stringify({status:"error",message:"Offline"}));
        })
    ));
});

self.addEventListener("activate",event=>{
    event.waitUntil(
        caches.keys().then(cacheNames=>Promise.all(
            cacheNames.filter(name=>name!=="vial-mcp-cache-v1").map(name=>caches.delete(name))
        ))
    );
    console.log("Service worker activated");
});
