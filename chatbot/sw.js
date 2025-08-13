self.addEventListener('install', event => {
    event.waitUntil(self.skipWaiting());
    console.log('Service Worker installed');
});

self.addEventListener('activate', event => {
    event.waitUntil(self.clients.claim());
    console.log('Service Worker activated');
});

const API_BASE = 'http://localhost:8000/api';
const BACKEND_NODES = [
    'http://localhost:8000/api',
    'http://localhost:8001/api',
    'http://localhost:8002/api',
    'http://localhost:8003/api',
    'http://localhost:8004/api',
    'http://localhost:8005/api'
];

async function tryNodes(endpoint, options) {
    for (const node of BACKEND_NODES) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            const response = await fetch(`${node}${endpoint}`, {
                ...options,
                headers: {
                    ...options.headers,
                    'Content-Type': 'application/json'
                },
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            if (response.ok) {
                console.log(`Connected to ${node}${endpoint}`);
                return response;
            }
            const errorText = await response.text().catch(() => 'No response body');
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        } catch (error) {
            console.log(`Node ${node}${endpoint} failed: ${error.message}`);
            self.clients.matchAll().then(clients => {
                clients.forEach(client => client.postMessage({
                    type: 'error',
                    message: `Node ${node}${endpoint} failed: ${error.message}`,
                    endpoint
                }));
            });
        }
    }
    console.log(`All nodes failed for ${endpoint}, returning mock response`);
    return new Response(JSON.stringify(getMockResponse(endpoint)), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
    });
}

self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            tryNodes(url.pathname, {
                method: event.request.method,
                headers: new Headers(event.request.headers),
                body: event.request.method !== 'GET' ? event.request.body : undefined
            }).catch(error => {
                console.error('Service Worker fetch failed:', error.message);
                self.clients.matchAll().then(clients => {
                    clients.forEach(client => client.postMessage({
                        type: 'error',
                        message: `Service Worker fetch failed: ${error.message}`,
                        endpoint: url.pathname
                    }));
                });
                return new Response(JSON.stringify(getMockResponse(url.pathname)), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' }
                });
            })
        );
    }
});

function getMockResponse(pathname) {
    const mockResponses = {
        '/api/health': { status: 'healthy', mongo: true, version: '2.8', services: ['auth', 'wallet', 'vials'] },
        '/api/auth/login': { apiKey: `api-${crypto.randomUUID()}`, walletAddress: 'mock-wallet', walletHash: 'mock-hash', userId: `user-${crypto.randomUUID()}` },
        '/api/auth/api-key/generate': { apiKey: `api-${crypto.randomUUID()}`, walletAddress: 'mock-wallet', walletHash: 'mock-hash' },
        '/api/vials/vial1/prompt': { response: 'Prompt processed for vial1' },
        '/api/vials/vial2/prompt': { response: 'Prompt processed for vial2' },
        '/api/vials/vial3/prompt': { response: 'Prompt processed for vial3' },
        '/api/vials/vial4/prompt': { response: 'Prompt processed for vial4' },
        '/api/vials/vial1/task': { status: 'Task assigned to vial1' },
        '/api/vials/vial2/task': { status: 'Task assigned to vial2' },
        '/api/vials/vial3/task': { status: 'Task assigned to vial3' },
        '/api/vials/vial4/task': { status: 'Task assigned to vial4' },
        '/api/vials/vial1/config': { status: 'Config updated for vial1' },
        '/api/vials/vial2/config': { status: 'Config updated for vial2' },
        '/api/vials/vial3/config': { status: 'Config updated for vial3' },
        '/api/vials/vial4/config': { status: 'Config updated
