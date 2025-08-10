```javascript
// Global state
let isAuthenticated = false;
let masterKey = null;
let walletKey = null;
let agenticNetworkId = null;
let vials = Array(4).fill().map((_, i) => ({
    id: `vial${i+1}`,
    status: 'stopped',
    code: 'import torch\nimport torch.nn as nn\n\nclass VialAgent(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 1)\n    def forward(self, x):\n        return torch.sigmoid(self.fc(x))\n\nmodel = VialAgent()',
    codeLength: 0,
    isPython: true,
    webxosHash: generateUUID(),
    wallet: { address: null, balance: 0 },
    tasks: []
}));
let wallet = { address: null, balance: 0 };
let db = null;
const serverUrl = '/api';
let sessionStartTime = null;

// UUID generator
function generateUUID() {
    try {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
            const r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
    } catch (err) {
        logError(`UUID Generation Error: ${err.message}`, 'Check Math.random', err.stack, 'HIGH', 'Script Errors');
        return '';
    }
}

// Initialize Dexie
async function initDexie() {
    try {
        if (!window.indexedDB) {
            logError('IndexedDB Not Supported', 'Using localStorage', 'No stack', 'HIGH', 'Script Errors');
            return;
        }
        if (typeof Dexie === 'undefined') {
            logError('Dexie Not Loaded', 'Check CDN or /static/dexie.min.js', 'No stack', 'CRITICAL', 'Script Errors');
            return;
        }
        db = new Dexie('WebXOSVial');
        db.version(1).stores({
            logs: '++id,timestamp,event_type,message,metadata,urgency',
            vials: 'id,status,code,codeLength,isPython,webxosHash,wallet,tasks',
            wallets: 'address,balance',
            errors: '++id,timestamp,message,analysis,stack,urgency,category',
            auth: 'key,token,timestamp',
            network: 'networkId'
        });
        await db.open();
        window.db = db;
        logEvent('system', 'Dexie 4.0.11 initialized', {}, 'LOW');
        const storedVials = await db.vials.toArray();
        if (storedVials.length === 4) vials = storedVials;
        const storedWallet = await db.wallets.get({ address: wallet.address || 'default' });
        if (storedWallet) wallet = storedWallet;
        const storedNetwork = await db.network.get('networkId');
        if (storedNetwork) agenticNetworkId = storedNetwork.networkId;
    } catch (err) {
        logError(`Dexie Init Error: ${err.message}`, 'Using localStorage, check Dexie CDN or /static/dexie.min.js', err.stack, 'CRITICAL', 'Script Errors');
        db = null;
    }
}

// Backup to local DB
async function backupToLocalDB() {
    try {
        if (!db) {
            logError('Backup Failed', 'Dexie not initialized, using localStorage', 'No stack', 'HIGH', 'Script Errors');
            localStorage.setItem('vials', JSON.stringify(vials));
            localStorage.setItem('wallet', JSON.stringify(wallet));
            localStorage.setItem('network', JSON.stringify({ networkId: agenticNetworkId }));
            return;
        }
        await db.vials.clear().then(() => db.vials.bulkPut(vials));
        await db.wallets.put(wallet);
        await db.network.put({ networkId: agenticNetworkId });
        logEvent('backup', 'Vials, wallet, and network backed up to local DB', {}, 'INFO');
    } catch (err) {
        logError(`Backup Error: ${err.message}`, 'Check Dexie or localStorage', err.stack, 'HIGH', 'Script Errors');
    }
}

// Compute $WEBXOS hash
async function computeWebxosHash(code, networkId) {
    try {
        const msgBuffer = new TextEncoder().encode(code + networkId);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    } catch (err) {
        logError(`Hash Computation Error: ${err.message}`, 'Check crypto.subtle', err.stack, 'HIGH', 'Script Errors');
        return generateUUID();
    }
}

// Parse imported .md file
async function parseImportedMD(file) {
    try {
        const text = await file.text();
        const lines = text.split('\n');
        let currentVial = null;
        let newVials = [];
        let newWallet = { address: null, balance: 0 };
        let newNetworkId = null;
        let inCodeBlock = false;
        let codeBlock = [];
        for (let line of lines) {
            if (line.startsWith('## Agentic Network')) {
                continue;
            } else if (line.startsWith('- Network ID: ')) {
                newNetworkId = line.replace('- Network ID: ', '').trim();
            } else if (line.startsWith('## Wallet')) {
                continue;
            } else if (line.startsWith('- Wallet Key: ')) {
                walletKey = line.replace('- Wallet Key: ', '').trim();
            } else if (line.startsWith('- Session Balance: ')) {
                newWallet.balance = parseFloat(line.replace('- Session Balance: ', '').replace(' $WEBXOS', '').trim());
            } else if (line.startsWith('- Address: ')) {
                newWallet.address = line.replace('- Address: ', '').trim();
            } else if (line.startsWith('# Vial Agent: vial')) {
                if (currentVial) {
                    newVials.push(currentVial);
                }
                currentVial = {
                    id: line.replace('# Vial Agent: ', '').trim(),
                    status: 'stopped',
                    code: '',
                    codeLength: 0,
                    isPython: true,
                    webxosHash: generateUUID(),
                    wallet: { address: null, balance: 0 },
                    tasks: []
                };
            } else if (line.startsWith('- Status: ')) {
                currentVial.status = line.replace('- Status: ', '').trim();
            } else if (line.startsWith('- Language: ')) {
                currentVial.isPython = line.replace('- Language: ', '').trim() === 'Python';
            } else if (line.startsWith('- Code Length: ')) {
                currentVial.codeLength = parseInt(line.replace('- Code Length: ', '').replace(' bytes', '').trim());
            } else if (line.startsWith('- $WEBXOS Hash: ')) {
                currentVial.webxosHash = line.replace('- $WEBXOS Hash: ', '').trim();
            } else if (line.startsWith('- Wallet Balance: ')) {
                currentVial.wallet.balance = parseFloat(line.replace('- Wallet Balance: ', '').replace(' $WEBXOS', '').trim());
            } else if (line.startsWith('- Wallet Address: ')) {
                currentVial.wallet.address = line.replace('- Wallet Address: ', '').trim();
            } else if (line.startsWith('- Tasks: ')) {
                currentVial.tasks = line.replace('- Tasks: ', '').trim().split(',').map(t => t.trim()).filter(t => t);
            } else if (line.startsWith('```python') || line.startsWith('```javascript')) {
                inCodeBlock = true;
                codeBlock = [];
            } else if (line.startsWith('```') && inCodeBlock) {
                inCodeBlock = false;
                currentVial.code = codeBlock.join('\n');
                const computedHash = await computeWebxosHash(currentVial.code, newNetworkId);
                if (computedHash !== currentVial.webxosHash) {
                    logError('Hash Mismatch', `Invalid $WEBXOS hash for ${currentVial.id}`, 'No stack', 'HIGH', 'Import Errors');
                    return;
                }
            } else if (inCodeBlock) {
                codeBlock.push(line);
            }
        }
        if (currentVial) {
            newVials.push(currentVial);
        }
        if (newVials.length === 4) {
            vials = newVials;
            wallet = newWallet;
            agenticNetworkId = newNetworkId;
            if (db) {
                db.vials.clear().then(() => db.vials.bulkPut(vials));
                db.wallets.put(wallet);
                db.network.put({ networkId: agenticNetworkId });
            } else {
                localStorage.setItem('vials', JSON.stringify(vials));
                localStorage.setItem('wallet', JSON.stringify(wallet));
                localStorage.setItem('network', JSON.stringify({ networkId: agenticNetworkId }));
            }
            logEvent('import', 'Imported agentic network from .md', { networkId: agenticNetworkId }, 'INFO');
            updateVialStatsUI();
            updateVialStatusBars();
            updateBalanceDisplay();
        } else {
            logError('Import Error', 'Invalid .md format, expected 4 vials', 'No stack', 'HIGH', 'Import Errors');
        }
    } catch (err) {
        logError(`Import Error: ${err.message}`, 'Check .md file format', err.stack, 'HIGH', 'Import Errors');
    }
}

// Reconstruct /vial/
async function reconstructVialFolder() {
    try {
        const response = await fetch('/vial/server.py');
        if (!response.ok) throw new Error('Vial folder missing');
        logEvent('system', 'Vial folder detected', {}, 'INFO');
    } catch (err) {
        logError(`Vial Folder Missing: ${err.message}`, 'Reconstructing /vial/ client-side', err.stack, 'HIGH', 'Script Errors');
        localStorage.setItem('vialStructure', JSON.stringify({
            'server.py': '# Mock server.py with PyTorch',
            'vial_manager.py': '# Mock vial_manager.py with PyTorch'
        }));
    }
}

// Check server connection
async function checkServerConnection() {
    try {
        if (typeof redaxios === 'undefined') {
            logEvent('system', 'Redaxios not loaded, simulating offline server response', {}, 'INFO');
            return { success: false, serverUrl: null, mockResponse: { status: 200, data: { vials, balance: 0 } } };
        }
        const response = await redaxios.get(`${serverUrl}/mcp/ping`);
        if (response.status === 200) {
            logEvent('system', `Server ping successful: ${serverUrl}`, {}, 'INFO');
            return { success: true, serverUrl };
        }
        throw new Error(`Server ping failed: ${response.status}`);
    } catch (error) {
        logError(`Connection Error: ${error.message}`, 'Simulating offline server response', error.stack || 'No stack', 'HIGH', 'Docker Errors');
        return { success: false, serverUrl: null, mockResponse: { status: 200, data: { vials, balance: 0 } } };
    }
}

// Verify master link
async function verifyMasterLink() {
    try {
        if (!isAuthenticated) {
            logError('Master link verification failed', 'Not authenticated', 'No stack', 'HIGH', 'Authentication Errors');
            return false;
        }
        if (masterKey === 'offline') {
            logEvent('system', 'Master link verified in offline mode', {}, 'INFO');
            await backupToLocalDB();
            return true;
        }
        const { success } = await checkServerConnection();
        if (!success) {
            logError('Master link verification failed', 'Re-authenticate to continue', 'No stack', 'HIGH', 'Authentication Errors');
            return false;
        }
        return true;
    } catch (err) {
        logError(`Verify Master Link Error: ${err.message}`, 'Check verifyMasterLink', err.stack, 'HIGH', 'Script Errors');
        return false;
    }
}

// Authentication
async function authenticate() {
    try {
        walletKey = generateUUID();
        agenticNetworkId = generateUUID();
        sessionStartTime = new Date().toISOString();
        const { success, serverUrl, mockResponse } = await checkServerConnection();
        if (!success) {
            isAuthenticated = true;
            masterKey = 'offline';
            wallet.address = generateUUID();
            vials.forEach(vial => {
                vial.wallet.address = generateUUID();
                vial.wallet.balance = wallet.balance / 4;
            });
            if (db) {
                db.auth.put({ key: 'master', token: 'offline', timestamp: Date.now() });
                db.wallets.put(wallet);
                db.vials.clear().then(() => db.vials.bulkPut(vials));
                db.network.put({ networkId: agenticNetworkId });
            } else {
                localStorage.setItem('auth', JSON.stringify({ key: 'master', token: 'offline', timestamp: Date.now() }));
                localStorage.setItem('wallet', JSON.stringify(wallet));
                localStorage.setItem('vials', JSON.stringify(vials));
                localStorage.setItem('network', JSON.stringify({ networkId: agenticNetworkId }));
            }
            document.getElementById('authButton').classList.add('active-monitor');
            ['trainButton', 'exportButton', 'uploadButton'].forEach(id => document.getElementById(id).disabled = false);
            logEvent('auth', 'Offline mode enabled with agentic network', { networkId: agenticNetworkId }, 'INFO');
            updateVialStatsUI();
            updateVialStatusBars();
            updateBalanceDisplay();
            await backupToLocalDB();
            return;
        }
        if (typeof redaxios === 'undefined') {
            logEvent('system', 'Redaxios not loaded, switching to offline mode', {}, 'INFO');
            isAuthenticated = true;
            masterKey = 'offline';
            wallet.address = generateUUID();
            vials.forEach(vial => {
                vial.wallet.address = generateUUID();
                vial.wallet.balance = wallet.balance / 4;
            });
            if (db) {
                db.auth.put({ key: 'master', token: 'offline', timestamp: Date.now() });
                db.wallets.put(wallet);
                db.vials.clear().then(() => db.vials.bulkPut(vials));
                db.network.put({ networkId: agenticNetworkId });
            } else {
                localStorage.setItem('auth', JSON.stringify({ key: 'master', token: 'offline', timestamp: Date.now() }));
                localStorage.setItem('wallet', JSON.stringify(wallet));
                localStorage.setItem('vials', JSON.stringify(vials));
                localStorage.setItem('network', JSON.stringify({ networkId: agenticNetworkId }));
            }
            document.getElementById('authButton').classList.add('active-monitor');
            ['trainButton', 'exportButton', 'uploadButton'].forEach(id => document.getElementById(id).disabled = false);
            logEvent('auth', 'Offline mode enabled with agentic network', { networkId: agenticNetworkId }, 'INFO');
            updateVialStatsUI();
            updateVialStatusBars();
            updateBalanceDisplay();
            await backupToLocalDB();
            return;
        }
        const response = await redaxios.post(`${serverUrl}/mcp/auth`, {
            client: 'vial',
            deviceId: generateUUID(),
            sessionId: generateUUID(),
            networkId: agenticNetworkId
        }, {
            headers: { 'Content-Type': 'application/json' }
        });
        if (response.status !== 200) {
            throw new Error(`Authentication failed: ${response.status}, Response: ${response.data.slice(0, 50)}`);
        }
        const { token, address } = response.data;
        if (token) {
            masterKey = token;
            isAuthenticated = true;
            wallet.address = address;
            wallet.balance = 0;
            vials.forEach(vial => {
                vial.wallet.address = generateUUID();
                vial.wallet.balance = wallet.balance / 4;
            });
            if (db) {
                db.auth.put({ key: 'master', token, timestamp: Date.now() });
                db.wallets.put(wallet);
                db.vials.clear().then(() => db.vials.bulkPut(vials));
                db.network.put({ networkId: agenticNetworkId });
            } else {
                localStorage.setItem('auth', JSON.stringify({ key: 'master', token, timestamp: Date.now() }));
                localStorage.setItem('wallet', JSON.stringify(wallet));
                localStorage.setItem('vials', JSON.stringify(vials));
                localStorage.setItem('network', JSON.stringify({ networkId: agenticNetworkId }));
            }
            document.getElementById('authButton').classList.add('active-monitor');
            ['trainButton', 'exportButton', 'uploadButton'].forEach(id => document.getElementById(id).disabled = false);
            logEvent('auth', 'Authentication successful. 4 vials allocated in agentic network.', { networkId: agenticNetworkId }, 'INFO');
            updateVialStatsUI();
            updateVialStatusBars();
            updateBalanceDisplay();
        }
    } catch (error) {
        logError(`Authentication failed: ${error.message}`, 'Check OAuth endpoint or server response', error.stack || 'No stack', 'HIGH', 'Authentication Errors');
    }
}

// Void
async function voidVials() {
    try {
        if (masterKey !== 'offline') {
            if (typeof redaxios === 'undefined') {
                logEvent('system', 'Redaxios not loaded, skipping server void in offline mode', {}, 'INFO');
            } else {
                const response = await redaxios.post(`${serverUrl}/mcp/void`, { networkId: agenticNetworkId }, {
                    headers: { 'Authorization': `Bearer ${masterKey}` }
                });
                if (response.status !== 200) throw new Error(`Void failed: ${response.status}`);
            }
        }
        if (db) {
            db.vials.clear();
            db.auth.clear();
            db.wallets.clear();
            db.network.clear();
        } else {
            localStorage.removeItem('vials');
            localStorage.removeItem('auth');
            localStorage.removeItem('wallet');
            localStorage.removeItem('network');
        }
        isAuthenticated = false;
        masterKey = null;
        walletKey = null;
        agenticNetworkId = null;
        wallet = { address: null, balance: 0 };
        sessionStartTime = null;
        vials = Array(4).fill().map((_, i) => ({
            id: `vial${i+1}`,
            status: 'stopped',
            code: 'import torch\nimport torch.nn as nn\n\nclass VialAgent(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 1)\n    def forward(self, x):\n        return torch.sigmoid(self.fc(x))\n\nmodel = VialAgent()',
            codeLength: 0,
            isPython: true,
            webxosHash: generateUUID(),
            wallet: { address: null, balance: 0 },
            tasks: []
        }));
        document.getElementById('authButton').classList.remove('active-monitor');
        ['trainButton', 'exportButton', 'uploadButton'].forEach(id => document.getElementById(id).disabled = true);
        logQueue = ['<p>Vial MCP Controller initialized</p>', '<p class="balance">$WEBXOS Balance: 0.0000</p>'];
        debouncedUpdateConsole();
        updateVialStatsUI();
        updateVialStatusBars();
        updateBalanceDisplay();
        logEvent('void', 'All data voided', {}, 'INFO');
    } catch (err) {
        logError(`Void Error: ${err.message}`, 'Check void function', err.stack || 'No stack', 'HIGH', 'Script Errors');
    }
}

// Troubleshoot
async function troubleshoot() {
    try {
        const { success } = await checkServerConnection();
        const authRecord = db ? await db.auth.get('master') : JSON.parse(localStorage.getItem('auth') || '{}');
        const vialsCount = db ? await db.vials.count() : (JSON.parse(localStorage.getItem('vials') || '[]').length);
        const network = db ? await db.network.get('networkId') : JSON.parse(localStorage.getItem('network') || '{}');
        logEvent('diagnostics', `Troubleshoot: Server ${success ? 'online' : 'offline'}, Auth ${authRecord?.key ? 'present' : 'missing'}, Vials: ${vialsCount}, Network: ${network?.networkId || 'none'}`, {}, 'INFO');
        await reconstructVialFolder();
    } catch (err) {
        logError(`Troubleshoot Error: ${err.message}`, 'Check diagnostics function', err.stack || 'No stack', 'HIGH', 'Script Errors');
    }
}

// Train Vials
async function trainVials() {
    try {
        if (!(await verifyMasterLink())) return;
        const fileInput = document.getElementById('file-input');
        if (!fileInput?.files?.length) {
            logError('Train Error', 'No file selected for training', 'No stack', 'HIGH', 'Script Errors');
            return;
        }
        const file = fileInput.files[0];
        if (!['.js', '.py', '.txt', '.md'].includes(file.name.slice(-4))) {
            logError('File Error', 'Only .js, .py, .txt, or .md allowed', 'No stack', 'HIGH', 'Script Errors');
            return;
        }
        if (file.size > 1024 * 1024) {
            logError('File Error', 'File size exceeds 1MB', 'No stack', 'HIGH', 'Script Errors');
            return;
        }
        const inputData = await file.text();
        const isPython = file.name.endsWith('.py');
        document.getElementById('trainButton').classList.add('active-train');
        document.getElementById('console').classList.add('active-train');
        document.body.classList.add('train-glow');
        logEvent('training', 'Training started', { networkId: agenticNetworkId }, 'INFO');
        const startTime = performance.now();
        let response = { status: 200, data: { vials, balance: 0 } };
        if (masterKey !== 'offline') {
            if (typeof redaxios === 'undefined') {
                logEvent('system', 'Redaxios not loaded, simulating training in offline mode', {}, 'INFO');
                response = { status: 200, data: { vials: vials.map(v => ({
                    ...v,
                    tasks: ['search_docs', 'read_emails', 'send_gmails', 'search_web'].filter(t => inputData.toLowerCase().includes(t))
                })), balance: 0.0004 } };
            } else {
                const formData = new FormData();
                formData.append('code', inputData);
                formData.append('isPython', isPython);
                formData.append('networkId', agenticNetworkId);
                response = await redaxios.post(`${serverUrl}/mcp/train`, formData, {
                    headers: { 'Authorization': `Bearer ${masterKey}` }
                });
                if (response.status !== 200) throw new Error(`Training failed: ${response.status}`);
            }
        } else {
            response = { status: 200, data: { vials: vials.map(v => ({
                ...v,
                tasks: ['search_docs', 'read_emails', 'send_gmails', 'search_web'].filter(t => inputData.toLowerCase().includes(t))
            })), balance: 0.0004 } };
        }
        const { vials: updatedVials, balance } = response.data;
        const trainingTime = (performance.now() - startTime) / 1000;
        wallet.balance += balance;
        vials.forEach(vial => {
            const updatedVial = updatedVials.find(v => v.id === vial.id) || {
                id: vial.id,
                tasks: ['search_docs', 'read_emails', 'send_gmails', 'search_web'].filter(t => inputData.toLowerCase().includes(t))
            };
            vial.code = inputData;
            vial.codeLength = inputData.length;
            vial.isPython = isPython;
            vial.status = 'running';
            vial.wallet.balance = wallet.balance / 4;
            vial.tasks = updatedVial.tasks;
            vial.webxosHash = computeWebxosHash(inputData, agenticNetworkId);
            if (db) db.vials.put(vial);
            else localStorage.setItem('vials', JSON.stringify(vials));
            logEvent('training', `Trained vial ${vial.id} with tasks ${vial.tasks.join(', ')}`, { webxosHash: vial.webxosHash }, 'INFO');
        });
        if (db) db.wallets.put(wallet);
        else localStorage.setItem('wallet', JSON.stringify(wallet));
        document.getElementById('trainButton').classList.remove('active-train');
        document.getElementById('console').classList.remove('active-train');
        document.body.classList.remove('train-glow');
        logEvent('training', `Training completed. Earned ${balance.toFixed(4)} $WEBXOS`, { networkId: agenticNetworkId }, 'INFO');
        updateVialStatsUI();
        updateVialStatusBars();
        updateBalanceDisplay();
        await backupToLocalDB();
    } catch (err) {
        document.getElementById('trainButton').classList.remove('active-train');
        document.getElementById('console').classList.remove('active-train');
        document.body.classList.remove('train-glow');
        logError(`Train Error: ${err.message}`, 'Check training function', err.stack || 'No stack', 'HIGH', 'Script Errors');
    }
}

// Export vials and wallet
async function exportVials() {
    try {
        if (!(await verifyMasterLink())) return;
        const sessionDuration = sessionStartTime ? ((new Date() - new Date(sessionStartTime)) / 1000).toFixed(2) : '0.00';
        const content = `# WebXOS Vial and Wallet Export\n\n## Agentic Network\n- Network ID: ${agenticNetworkId || 'none'}\n- Session Start: ${sessionStartTime || 'none'}\n- Session Duration: ${sessionDuration} seconds\n\n## Wallet\n- Wallet Key: ${walletKey || 'none'}\n- Session Balance: ${wallet.balance.toFixed(4)} $WEBXOS\n- Address: ${wallet.address || 'offline'}\n\n## Vials\n${vials.map(vial => `# Vial Agent: ${vial.id}\n- Status: ${vial.status}\n- Language: ${vial.isPython ? 'Python' : 'JavaScript'}\n- Code Length: ${vial.codeLength} bytes\n- $WEBXOS Hash: ${vial.webxosHash}\n- Wallet Balance: ${vial.wallet.balance.toFixed(4)} $WEBXOS\n- Wallet Address: ${vial.wallet.address || 'none'}\n- Tasks: ${vial.tasks.join(', ') || 'none'}\n\n\`\`\`${vial.isPython ? 'python' : 'javascript'}\n${vial.code}\n\`\`\`\n`).join('---\n\n')}\n## Instructions\n- **Reuse**: Import this .md file via the "Upload" button in Vial MCP Controller to resume training.\n- **Extend**: Modify agent code in external projects, then reimport.\n- **Cash Out**: $WEBXOS balance is tied to the wallet address for future Stripe integration.\n\nGenerated by Vial MCP Controller`;
        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vial_wallet_export_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
        a.click();
        URL.revokeObjectURL(url);
        logEvent('export', 'Exported vials and wallet as Markdown', { networkId: agenticNetworkId }, 'INFO');
    } catch (err) {
        logError(`Export Error: ${err.message}`, 'Check export function', err.stack || 'No stack', 'HIGH', 'Script Errors');
    }
}

// Upload file
async function uploadFile() {
    try {
        if (!(await verifyMasterLink())) return;
        const fileInput = document.getElementById('file-input');
        if (!fileInput?.files?.length) {
            logError('Upload Error', 'No file selected', 'No stack', 'HIGH', 'Script Errors');
            return;
        }
        const file = fileInput.files[0];
        if (!file.name.endsWith('.js') && !file.name.endsWith('.py') && !file.name.endsWith('.txt') && !file.name.endsWith('.md')) {
            logError('Upload Error', 'Only .js, .py, .txt, or .md allowed', 'No stack', 'HIGH', 'Script Errors');
            return;
        }
        if (file.size > 1024 * 1024) {
            logError('Upload Error', 'File size exceeds 1MB', 'No stack', 'HIGH', 'Script Errors');
            return;
        }
        if (file.name.endsWith('.md')) {
            await parseImportedMD(file);
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        formData.append('networkId', agenticNetworkId);
        let response = { status: 200, data: { filePath: '/uploads/mock' } };
        if (masterKey !== 'offline') {
            if (typeof redaxios === 'undefined') {
                logEvent('system', 'Redaxios not loaded, simulating upload in offline mode', {}, 'INFO');
            } else {
                response = await redaxios.post(`${serverUrl}/mcp/upload`, formData, {
                    headers: { 'Authorization': `Bearer ${masterKey}` }
                });
                if (response.status !== 200) throw new Error(`Upload failed: ${response.status}`);
            }
        }
        const { filePath } = response.data;
        logEvent('upload', `File uploaded to ${filePath}`, { networkId: agenticNetworkId }, 'INFO');
        fileInput.value = '';
        await backupToLocalDB();
    } catch (err) {
        logError(`Upload Error: ${err.message}`, 'Check upload function', err.stack || 'No stack', 'HIGH', 'Script Errors');
    }
}

// Update vial stats UI
function updateVialStatsUI() {
    try {
        document.getElementById('vial-stats').innerHTML = vials.map(vial => `
            <div class="progress-container">
                <span class="progress-label">${vial.id}</span>
                <div class="progress-bar">
                    <div class="progress-fill ${vial.status === 'stopped' ? 'offline' : ''}" style="width: ${vial.status === 'running' ? '100%' : '0%'};"></div>
                </div>
                <span>${vial.status} | ${vial.wallet.balance.toFixed(4)}</span>
            </div>
        `).join('');
    } catch (err) {
        logError(`Vial Stats UI Error: ${err.message}`, 'Check updateVialStatsUI', err.stack, 'HIGH', 'Script Errors');
    }
}

// Update vial status bars
function updateVialStatusBars() {
    try {
        document.getElementById('vial-status-bars').innerHTML = vials.map(vial => `
            <div class="progress-container">
                <span class="progress-label">${vial.id}</span>
                <div class="progress-bar">
                    <div class="progress-fill ${vial.status === 'stopped' ? 'offline' : ''}" style="width: ${vial.status === 'running' ? '100%' : '0%'};"></div>
                </div>
                <span class="status-text">${vial.status} | ${vial.codeLength} bytes | ${vial.tasks.join(', ') || 'none'}</span>
            </div>
        `).join('');
    } catch (err) {
        logError(`Vial Status Bars Error: ${err.message}`, 'Check updateVialStatusBars', err.stack, 'HIGH', 'Script Errors');
    }
}

// Update balance display
function updateBalanceDisplay() {
    try {
        const balanceIndex = logQueue.findIndex(log => log.includes('$WEBXOS Balance'));
        if (balanceIndex !== -1) {
            logQueue[balanceIndex] = `<p class="balance">$WEBXOS Balance: ${wallet.balance.toFixed(4)}</p>`;
        } else {
            logQueue.push(`<p class="balance">$WEBXOS Balance: ${wallet.balance.toFixed(4)}</p>`);
        }
        if (logQueue.length > 50) logQueue.shift();
        debouncedUpdateConsole();
    } catch (err) {
        logError(`Balance Display Error: ${err.message}`, 'Check updateBalanceDisplay', err.stack, 'HIGH', 'Script Errors');
    }
}

// Initialize
try {
    initDexie();
    reconstructVialFolder();
    updateVialStatusBars();
    updateBalanceDisplay();
} catch (err) {
    logError(`Initialization Error: ${err.message}`, 'Check initDexie or reconstructVialFolder', err.stack, 'CRITICAL', 'Script Errors');
}

// Event listeners
try {
    document.getElementById('authButton').addEventListener('click', authenticate);
    document.getElementById('voidButton').addEventListener('click', voidVials);
    document.getElementById('troubleshootButton').addEventListener('click', troubleshoot);
    document.getElementById('trainButton').addEventListener('click', trainVials);
    document.getElementById('exportButton').addEventListener('click', exportVials);
    document.getElementById('uploadButton').addEventListener('click', () => document.getElementById('file-input').click());
    document.getElementById('file-input').addEventListener('change', () => {
        logEvent('file', 'File selected', { networkId: agenticNetworkId }, 'INFO');
        uploadFile();
    });
} catch (err) {
    logError(`Event Listener Error: ${err.message}`, 'Check DOM elements or event bindings', err.stack, 'CRITICAL', 'Script Errors');
}
```
