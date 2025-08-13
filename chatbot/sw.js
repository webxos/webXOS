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
    'http://localhost:8002/api'
];

async function tryNodes(endpoint, options) {
    for (const node of BACKEND_NODES) {
        try {
            const response = await fetch(`${node}${endpoint}`, {
                ...options,
                headers: {
                    ...options.headers,
                    'Content-Type': 'application/json'
                }
            });
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
        '/api/vials/vial4/config': { status: 'Config updated for vial4' },
        '/api/vials/void': { status: 'All vials reset' },
        '/api/wallet/create': { status: 'Wallet created', address: 'mock-wallet', webxos: 0.0 },
        '/api/wallet/import': { status: 'Wallet imported' },
        '/api/wallet/transaction': { status: 'Transaction recorded' },
        '/api/quantum/link': { statuses: ['running', 'running', 'running', 'running'], latencies: [50, 60, 70, 80] },
        '/api/blockchain/transaction': { status: 'Transaction recorded' }
    };
    return mockResponses[pathname] || { error: 'No mock response available' };
}
