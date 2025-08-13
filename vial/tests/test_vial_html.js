const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');

describe('Vial HTML Tests', () => {
    let dom;
    let window;
    let document;

    beforeEach(async () => {
        const html = fs.readFileSync(path.resolve(__dirname, '../../vial.html'), 'utf8');
        dom = new JSDOM(html, { runScripts: 'dangerously', resources: 'usable' });
        window = dom.window;
        document = window.document;

        // Mock Dexie
        window.Dexie = class {
            constructor(name) { this.name = name; }
            version(v) { return { stores: () => {} }; }
            table(name) { return { clear: async () => {}, put: async () => {} }; }
        };

        // Mock Redaxios
        window.redaxios = {
            get: async () => ({ data: { agents: { vial1: { status: 'running', wallet_balance: 100, wallet_address: 'addr1', wallet_hash: 'hash1', script: 'code' } } } }),
            post: async (url) => {
                if (url === '/api/auth') return { data: { apiKey: 'test_key' } };
                if (url === '/api/wallet/cashout') return { data: { status: 'success' } };
                if (url === '/api/import') return { data: { agents: { vial1: { status: 'running', wallet_balance: 100, wallet_address: 'addr1', wallet_hash: 'hash1', script: 'code' } } } };
            }
        };

        // Mock localStorage
        window.localStorage = { getItem: () => 'test_key', setItem: () => {} };
    });

    test('auth command triggers authentication', async () => {
        const input = document.getElementById('input');
        const output = document.getElementById('output');
        input.value = 'auth test_user';
        const event = new window.KeyboardEvent('keypress', { key: 'Enter' });
        input.dispatchEvent(event);
        await new Promise(resolve => setTimeout(resolve, 0));
        expect(output.textContent).toContain('Authenticated as test_user');
    });

    test('vials command loads vials', async () => {
        const input = document.getElementById('input');
        const output = document.getElementById('output');
        input.value = 'vials';
        const event = new window.KeyboardEvent('keypress', { key: 'Enter' });
        input.dispatchEvent(event);
        await new Promise(resolve => setTimeout(resolve, 0));
        expect(output.textContent).toContain('Loaded 1 vials');
    });

    test('cashout command triggers cashout', async () => {
        const input = document.getElementById('input');
        const output = document.getElementById('output');
        input.value = 'cashout 10 addr1';
        const event = new window.KeyboardEvent('keypress', { key: 'Enter' });
        input.dispatchEvent(event);
        await new Promise(resolve => setTimeout(resolve, 0));
        expect(output.textContent).toContain('Cashed out 10 $WEBXOS to addr1');
    });
});
