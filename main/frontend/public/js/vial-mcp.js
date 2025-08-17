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
let toolboxClient = null;
let toolboxTools = null;

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
        return null;
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
        const mode = vial.isTraining ? 'Training' : (isOffline ? 'Offline (Wallet Disabled)' : (toolboxClient ? 'GenAI' : 'Online'));
        const statusClass = vial.isTraining ? 'training' : (isOffline ? 'offline-grey' : (toolboxClient ? 'genai' : 'online'));
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
        updateBalanceDisplay();
    }, 10000);
}

// Disable functions on auth loss or offline mode
function disableFunctions() {
    const buttons = ['quantumLinkButton', 'exportButton', 'importButton', 'googleGenAIButton'];
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

// Initialize Google GenAI Toolbox client
async function initializeToolboxClient() {
    if (isOffline) {
        logEvent('error', 'Google GenAI Toolbox cannot be initialized in offline mode.', {});
        return;
    }
    if (!isAuthenticated) {
        logEvent('error', 'Authentication required to initialize Google GenAI Toolbox.', {});
        return;
    }
    try {
        const response = await axios.get('http://127.0.0.1:5000/mcp/toolsets', {
            headers: { 'Authorization': `Bearer ${apiCredentials.key}` }
        });
        toolboxTools = response.data.tools;
        toolboxClient = { url: 'http://127.0.0.1:5000', tools: toolboxTools };
        const googleGenAIButton = document.getElementById('googleGenAIButton');
        if (googleGenAIButton) {
            googleGenAIButton.disabled = false;
            googleGenAIButton.classList.add('active-genai');
        }
        document.body.classList.add('genai-glow');
        document.getElementById('console').classList.add('active-genai');
        logEvent('genai', 'Google GenAI Toolbox initialized. Available tools: ' + toolboxTools.map(t => t.name).join(', '), {});
        updateVialStatusBars();
    } catch (error) {
        logEvent('error', `Failed to initialize Google GenAI Toolbox: ${error.message}`, { error });
        showErrorNotification('Failed to connect to Google GenAI Toolbox server.');
    }
}

// Execute GenAI tool
async function executeGenAITool(toolName, params) {
    if (!toolboxClient) {
        logEvent('error', 'Google GenAI Toolbox not initialized.', {});
        return null;
    }
    try {
        const response = await axios.post(`${toolboxClient.url}/mcp/tools/${toolName}/execute`, { parameters: params }, {
            headers: { 'Authorization': `Bearer ${apiCredentials.key}` }
        });
        const blockHash = await addToBlockchain('genai_tool', { toolName, params, result: response.data });
        logEvent('genai', `Executed tool ${toolName}: ${JSON.stringify(response.data)} | Block: ${blockHash.slice(0, 8)}...`, { toolName, params });
        return response.data;
    } catch (error) {
        logEvent('error', `Failed to execute tool ${toolName}: ${error.message}`, { error });
        showErrorNotification(`Tool execution failed: ${error.message}`);
        return null;
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
    apiCredentials = { key: isOffline ? null : generateUUID(), secret: isOffline ? null : generateUUID() };
    isAuthenticated = true;
    await addToBlockchain('auth', { wallet: wallet.address, networkId: agenticNetworkId });
    vials.forEach((vial, i) => {
        vial.wallet = { address: isOffline ? null : generateUUID(), balance: 0, hash: wallet.hash };
        vial.quantumState = { qubits: [], entanglement: 'initialized' };
    });
    const authButton = document.getElementById('authButton');
    if (authButton) authButton.classList.add('active-monitor');
    ['quantumLinkButton', 'exportButton', 'importButton', 'googleGenAIButton'].forEach(id => {
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

// Handle Git and GenAI commands
async function handleGitCommand(input) {
    const sanitizedInput = sanitizeInput(input.trim());
    if (!sanitizedInput) return;
    logEvent('command', sanitizedInput);
    const parts = sanitizedInput.split(' ');
    const command = parts[0].toLowerCase();

    if (command === '/genai' && parts.length > 1) {
        if (!toolboxClient) {
            logEvent('error', 'Google GenAI Toolbox not initialized. Click "Google GenAI" button to initialize.', {});
            return;
        }
        const toolName = parts[1];
        const params = parts.slice(2);
        const tool = toolboxTools.find(t => t.name === toolName);
        if (!tool) {
            logEvent('error', `Tool ${toolName} not found. Available tools: ${toolboxTools.map(t => t.name).join(', ')}`, {});
            return;
        }
        const result = await executeGenAITool(toolName, params);
        if (result) {
            logEvent('genai', `Tool ${toolName} result: ${JSON.stringify(result)}`, { toolName, params });
        }
    } else if (command === '/help') {
        const helpText = `
            Available commands:
            /prompt vial[1-4] train <dataset> - Train a vial
            /genai <tool_name> <params> - Execute Google GenAI tool (e.g., /genai search-hotels-by-name Marriott)
            /status - Check vial statuses
            /auth - Re-authenticate
        `;
        logEvent('info', helpText.replace(/\n/g, '<br>'), {});
    } else {
        logEvent('error', `Unknown command: ${sanitizedInput}. Use /help for available commands.`, {});
    }
}

// Void vials
async function voidVials() {
    if (!isAuthenticated) {
        logEvent('error', 'Authentication required to void vials.', {});
        return;
    }
    vials.forEach(vial => {
        vial.status = 'stopped';
        vial.isTraining = false;
        vial.tasks = [];
        vial.trainingData = [];
    });
    await addToBlockchain('void', { vials: vials.map(v => v.id) });
    logEvent('system', 'All vials voided.', {});
    updateVialStatusBars();
}

// Troubleshoot
async function troubleshoot() {
    if (!isAuthenticated) {
        logEvent('error', 'Authentication required to troubleshoot.', {});
        return;
    }
    const issues = [];
    vials.forEach(vial => {
        if (vial.status === 'stopped' && vial.tasks.length > 0) {
            issues.push(`${vial.id}: Stopped but has pending tasks.`);
        }
        if (vial.latency > 200) {
            issues.push(`${vial.id}: High latency detected (${vial.latency}ms).`);
        }
    });
    if (toolboxClient) {
        try {
            await axios.get(`${toolboxClient.url}/health`, {
                headers: { 'Authorization': `Bearer ${apiCredentials.key}` }
            });
            logEvent('genai', 'Google GenAI Toolbox server health: OK', {});
        } catch (error) {
            issues.push(`Google GenAI Toolbox server: Unhealthy (${error.message})`);
        }
    }
    if (issues.length === 0) {
        logEvent('system', 'No issues detected.', {});
    } else {
        issues.forEach(issue => logEvent('error', issue, {}));
    }
    updateVialStatusBars();
}

// Quantum link
async function quantumLink() {
    if (!isAuthenticated) {
        logEvent('error', 'Authentication required for quantum link.', {});
        return;
    }
    vials.forEach(vial => {
        vial.quantumState = { qubits: [Math.random(), Math.random()], entanglement: 'linked' };
    });
    const blockHash = await addToBlockchain('quantum_link', { vials: vials.map(v => v.id) });
    logEvent('system', `Quantum link established. Block: ${blockHash.slice(0, 8)}...`, {});
}

// Export vials
async function exportVials() {
    if (!isAuthenticated) {
        logEvent('error', 'Authentication required to export vials.', {});
        return;
    }
    const exportData = {
        vials: vials.map(vial => ({
            id: vial.id,
            code: vial.code,
            status: vial.status,
            wallet: vial.wallet,
            quantumState: vial.quantumState
        })),
        blockchain: blockchain.slice(-10)
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'vial-mcp-export.json';
    a.click();
    URL.revokeObjectURL(url);
    logEvent('system', 'Vials exported successfully.', {});
}

// Import file
async function importFile(event) {
    if (!isAuthenticated) {
        logEvent('error', 'Authentication required to import files.', {});
        return;
    }
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (e) => {
        const content = e.target.result;
        if (!validateMarkdown(content)) {
            logEvent('error', 'Invalid .md file format.', {});
            return;
        }
        const { tasks, parameters } = parseMarkdownForTraining(content);
        vials.forEach(vial => {
            vial.tasks = tasks;
            vial.config = parameters;
        });
        const blockHash = await addToBlockchain('import', { tasks, parameters });
        logEvent('system', `Imported tasks and parameters. Block: ${blockHash.slice(0, 8)}...`, { tasks, parameters });
        updateVialStatusBars();
    };
    reader.readAsText(file);
}

// Show API popup
function showApiPopup() {
    if (!isAuthenticated || isOffline) {
        logEvent('error', 'Authentication required and online mode needed for API access.', {});
        return;
    }
    const apiInput = document.getElementById('api-input');
    const apiPopup = document.getElementById('api-popup');
    apiInput.value = JSON.stringify(apiCredentials, null, 2);
    apiPopup.classList.add('visible');
}

// Generate API credentials
async function generateApiCredentials() {
    if (!isAuthenticated || isOffline) {
        logEvent('error', 'Authentication required and online mode needed for API credentials.', {});
        return;
    }
    apiCredentials = { key: generateUUID(), secret: generateUUID() };
    const blockHash = await addToBlockchain('api_credentials', { key: apiCredentials.key });
    const apiInput = document.getElementById('api-input');
    apiInput.value = JSON.stringify(apiCredentials, null, 2);
    logEvent('system', `New API credentials generated. Block: ${blockHash.slice(0, 8)}...`, {});
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
        googleGenAIButton: document.getElementById('googleGenAIButton'),
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
    elements.googleGenAIButton.addEventListener('click', initializeToolboxClient);
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
