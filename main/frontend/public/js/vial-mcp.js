// Agent templates (simulating vial/agents/agent1.py to agent4.py)
const agentTemplates = [
    `import torch
import torch.nn as nn

class VialAgent1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = VialAgent1()`,
    `import torch
import torch.nn as nn

class VialAgent2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 2)
    def forward(self, x):
        return torch.relu(self.fc(x))

model = VialAgent2()`,
    `import torch
import torch.nn as nn

class VialAgent3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(15, 3)
    def forward(self, x):
        return torch.tanh(self.fc(x))

model = VialAgent3()`,
    `import torch
import torch.nn as nn

class VialAgent4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(25, 4)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

model = VialAgent4()`
];

// Global state
let logQueue = ['<p>Vial MCP Controller initialized. Use /help for API commands.</p>', '<p class="balance">$WEBXOS Balance: 0.0000 | Reputation: 0</p>'];
let isAuthenticated = false;
let isOffline = true;
let masterKey = null;
let walletKey = null;
let agenticNetworkId = null;
let tokenInterval = null;
let reputation = 0;
let blockchain = [];
let apiCredentials = { key: null, secret: null };
let vials = Array(4).fill().map((_, i) => ({
    id: `vial${i+1}`,
    status: 'stopped',
    code: agentTemplates[i],
    codeLength: agentTemplates[i].length,
    isPython: true,
    webxosHash: generateUUID(),
    wallet: { address: null, balance: 0, hash: null },
    tasks: [],
    quantumState: null,
    trainingData: [],
    config: {},
    isTraining: false,
    latency: 0
}));
let wallet = { address: null, balance: 0, hash: null };
let lastLogMessage = null;
let lastLogTime = 0;
let lastLogId = 0;

// UUID generator
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

// AES-256 encryption
async function encryptData(data) {
    const key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt']);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoded = new TextEncoder().encode(data);
    const encrypted = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, encoded);
    return { encrypted: Array.from(new Uint8Array(encrypted)), iv: Array.from(iv) };
}

// SHA-256 hash
async function sha256(data) {
    const encoded = new TextEncoder().encode(data);
    const hash = await crypto.subtle.digest('SHA-256', encoded);
    return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
}

// Sanitize input
function sanitizeInput(input) {
    return input.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
                .replace(/[<>{}]/g, '');
}

// Validate .md format
function validateMarkdown(mdContent) {
    if (!mdContent.includes('## Agentic Network') || mdContent.includes('<script')) {
        return false;
    }
    return true;
}

// Parse .md for training data
function parseMarkdownForTraining(mdContent) {
    const lines = mdContent.split('\n');
    let tasks = [];
    let parameters = {};
    let inTaskSection = false;
    for (let line of lines) {
        if (line.startsWith('## Tasks')) {
            inTaskSection = true;
        } else if (inTaskSection && line.startsWith('- ')) {
            tasks.push(line.slice(2).trim());
        } else if (line.match(/^- Parameter: (\w+): (.+)/)) {
            const [, key, value] = line.match(/^- Parameter: (\w+): (.+)/);
            parameters[key] = value;
        }
    }
    return { tasks, parameters };
}

// Simulate blockchain transaction
async function addToBlockchain(type, data) {
    if (isOffline && type !== 'train' && type !== 'import' && type !== 'export') {
        return null; // Skip blockchain in offline mode for non-training actions
    }
    const timestamp = new Date().toISOString();
    const prevHash = blockchain.length ? blockchain[blockchain.length - 1].hash : '0'.repeat(64);
    const hash = await sha256(`${type}${JSON.stringify(data, null, 0)}${timestamp}${prevHash}`);
    const block = { type, data, timestamp, prevHash, hash };
    blockchain.push(block);
    return hash;
}

// Debounce utility
function debounce(func, wait) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

// Log event with deduplication
function logEvent(event_type, message, metadata = {}) {
    const timestamp = new Date().toISOString();
    const now = Date.now();
    const logId = ++lastLogId;
    const baseMessage = message.replace(/^\[\S+\]\s*|\s*\[ID:\d+\]$/, '').trim();
    if (baseMessage === lastLogMessage && (now - lastLogTime) < 300) return;
    lastLogMessage = baseMessage;
    lastLogTime = now;
    const formattedMessage = `[${timestamp}] ${message} [ID:${logId}]`;
    logQueue.push(`<p class="${event_type === 'error' ? 'error' : 'command'}">${formattedMessage}</p>`);
    if (logQueue.length > 50) logQueue.shift();
    debouncedUpdateConsole();
    console.log(`${event_type}: ${message}`, metadata);
}

// Error notification
function showErrorNotification(message) {
    const errorNotification = document.getElementById('error-notification');
    if (errorNotification) {
        errorNotification.textContent = message;
        errorNotification.classList.add('visible');
        setTimeout(() => errorNotification.classList.remove('visible'), 5000);
    } else {
        console.error(`Notification: ${message}`);
    }
}

// Update console
const debouncedUpdateConsole = debounce(() => {
    const consoleDiv = document.getElementById('console');
    if (consoleDiv) {
        consoleDiv.innerHTML = logQueue.join('');
        consoleDiv.scrollTop = consoleDiv.scrollHeight;
    }
}, 100);

// Update vial status bars
function updateVialStatusBars() {
    const vialStatusBars = document.getElementById('vial-status-bars');
    if (!vialStatusBars) return;
    vials.forEach(vial => {
        vial.latency = isOffline ? 0 : Math.floor(Math.random() * (200 - 50 + 1)) + 50;
    });
    vialStatusBars.innerHTML = vials.map(vial => {
        const mode = vial.isTraining ? 'Training' : (isOffline ? 'Offline (Wallet Disabled)' : 'Online');
        const statusClass = vial.isTraining ? 'training' : (isOffline ? 'offline-grey' : 'online');
        const fillClass = vial.status === 'running' ? statusClass : '';
        return `
            <div class="progress-container">
                <span class="progress-label">${vial.id}</span>
                <div class="progress-bar">
                    <div class="progress-fill ${fillClass}" style="width: ${vial.status === 'running' ? '100%' : '0%'};"></div>
                </div>
                <span class="status-text ${statusClass}">Latency: ${vial.latency}ms | Size: ${vial.codeLength} bytes | Mode: ${mode}</span>
            </div>
        `;
    }).join('');
}

// Update balance and reputation display
function updateBalanceDisplay() {
    const balanceIndex = logQueue.findIndex(log => log.includes('$WEBXOS Balance'));
    const displayBalance = isOffline ? 'N/A (Offline)' : wallet.balance.toFixed(4);
    const displayReputation = isOffline ? 'N/A (Offline)' : reputation;
    if (balanceIndex !== -1) {
        logQueue[balanceIndex] = `<p class="balance">$WEBXOS Balance: ${displayBalance} | Reputation: ${displayReputation}</p>`;
    } else {
        logQueue.push(`<p class="balance">$WEBXOS Balance: ${displayBalance} | Reputation: ${displayReputation}</p>`);
    }
    if (logQueue.length > 50) logQueue.shift();
    debouncedUpdateConsole();
}

// Earn tokens and reputation
async function startTokenEarning() {
    if (tokenInterval) clearInterval(tokenInterval);
    if (isOffline) {
        logEvent('error', '$WEBXOS earning disabled in offline mode. Switch to online mode to earn tokens.', {});
        return;
    }
    tokenInterval = setInterval(async () => {
        if (!isAuthenticated || isOffline) {
            clearInterval(tokenInterval);
            disableFunctions();
            logEvent('error', 'Authentication lost or offline: $WEBXOS earning stopped.', {});
            return;
        }
        wallet.balance += 1;
        reputation += 1;
        const blockHash = await addToBlockchain('token_earn', { wallet: wallet.address, amount: 1, reputation });
        vials.forEach(vial => {
            vial.wallet.balance = wallet.balance / 4;
            vial.wallet.hash = blockHash;
        });
        wallet.hash = blockHash;
        logEvent('token', `Earned 1 $WEBXOS | Reputation: ${reputation} | Block: ${blockHash.slice(0, 8)}...`, { wallet: wallet.address });
        logEvent('token', `Earned 1 $WEBXOS | Reputation: ${reputation} | Block: ${blockHash.slice(0, 8)}...`, { wallet: wallet.address });
        updateBalanceDisplay();
    }, 10000);
}

// Disable functions on auth loss or offline mode
function disableFunctions() {
    const buttons = ['quantumLinkButton', 'exportButton', 'importButton'];
    if (isOffline) buttons.push('apiAccessButton');
    buttons.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = true;
    });
    const authButton = document.getElementById('authButton');
    if (authButton) authButton.classList.remove('active-monitor');
    if (isOffline) {
        clearInterval(tokenInterval);
        apiCredentials = { key: null, secret: null };
        logEvent('system', 'Offline mode: $WEBXOS earning and API access disabled.', {});
        updateVialStatusBars();
    }
}

// Authenticate
async function authenticate() {
    const isOnline = confirm('Authenticate in online mode? Cancel for offline mode.');
    isOffline = !isOnline;
    agenticNetworkId = generateUUID();
    masterKey = isOffline ? 'offline' : 'online';
    walletKey = isOffline ? null : generateUUID();
    wallet = { address: isOffline ? null : generateUUID(), balance: 0, hash: isOffline ? null : await sha256(agenticNetworkId) };
    reputation = 0;
    blockchain = [];
    apiCredentials = { key: null, secret: null };
    isAuthenticated = true;
    await addToBlockchain('auth', { wallet: wallet.address, networkId: agenticNetworkId });
    vials.forEach((vial, i) => {
        vial.wallet = { address: isOffline ? null : generateUUID(), balance: 0, hash: wallet.hash };
        vial.quantumState = { qubits: [], entanglement: 'initialized' };
    });
    const authButton = document.getElementById('authButton');
    if (authButton) authButton.classList.add('active-monitor');
    ['quantumLinkButton', 'exportButton', 'importButton'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = false;
    });
    if (!isOffline) {
        const apiAccessButton = document.getElementById('apiAccessButton');
        if (apiAccessButton) apiAccessButton.disabled = false;
    }
    logEvent('auth', `Authentication successful (${isOffline ? 'offline' : 'online'} mode). Quantum network allocated. Use /help for API commands.`, { networkId: agenticNetworkId });
    startTokenEarning();
    updateVialStatusBars();
    updateBalanceDisplay();
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateVialStatusBars();
    updateBalanceDisplay();
    const elements = {
        authButton: document.getElementById('authButton'),
        voidButton: document.getElementById('voidButton'),
        troubleshootButton: document.getElementById('troubleshootButton'),
        quantumLinkButton: document.getElementById('quantumLinkButton'),
        exportButton: document.getElementById('exportButton'),
        importButton: document.getElementById('importButton'),
        apiAccessButton: document.getElementById('apiAccessButton'),
        apiSubmit: document.getElementById('api-generate'),
        apiClose: document.getElementById('api-close'),
        fileInput: document.getElementById('file-input'),
        promptInput: document.getElementById('prompt-input')
    };

    elements.authButton.addEventListener('click', authenticate);
    elements.voidButton.addEventListener('click', voidVials);
    elements.troubleshootButton.addEventListener('click', troubleshoot);
    elements.quantumLinkButton.addEventListener('click', quantumLink);
    elements.exportButton.addEventListener('click', exportVials);
    elements.importButton.addEventListener('click', () => elements.fileInput.click());
    elements.apiAccessButton.addEventListener('click', showApiPopup);
    elements.apiSubmit.addEventListener('click', generateApiCredentials);
    elements.apiClose.addEventListener('click', () => {
        const apiPopup = document.getElementById('api-popup');
        if (apiPopup) apiPopup.classList.remove('visible');
    });
    elements.fileInput.addEventListener('change', importFile);
    elements.promptInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (elements.promptInput.value.trim()) {
                handleGitCommand(elements.promptInput.value);
                elements.promptInput.value = '';
            }
        }
    });

    logEvent('system', 'Vial MCP Controller initialized. Use /help for API commands.', {});
}, { once: true });
</script>
</body>
</html>
