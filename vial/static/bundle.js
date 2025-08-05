// bundle.js (source before esbuild)
import redaxios from '/static/redaxios.min.js';
import LZString from '/static/lz-string.min.js';
import Mustache from '/static/mustache.min.js';
import Dexie from '/static/dexie.min.js';

const API_BASE_URL = 'http://localhost:8080/mcp';
let monitorActive = false;
let trainActive = false;
let monitorInterval = null;
let trainTimeout = null;
let vials = [
    { name: 'vial1', latencyHistory: [], status: 'stopped', code: '', filePath: '', createdAt: '', codeLength: 0 },
    { name: 'vial2', latencyHistory: [], status: 'stopped', code: '', filePath: '', createdAt: '', codeLength: 0 },
    { name: 'vial3', latencyHistory: [], status: 'stopped', code: '', filePath: '', createdAt: '', codeLength: 0 },
    { name: 'vial4', latencyHistory: [], status: 'stopped', code: '', filePath: '', createdAt: '', codeLength: 0 }
];
let logQueue = [];
let errorLog = [];
const db = new Dexie('VialMCP');
db.version(1).stores({ logs: '++id,timestamp,event_type' });
const worker = new Worker('/static/worker.js');

worker.onmessage = (e) => {
    const { status, logs, message } = e.data;
    if (status === 'initialized') logEvent('system', 'SQLite initialized', { component: 'worker' }, 'LOW');
    else if (status === 'logs') updateConsole(logs);
    else if (status === 'error') logError(`Worker Error: ${message}`, 'Check worker.js', '', 'HIGH');
};
worker.postMessage({ action: 'init' });

async function logEvent(event_type, message, metadata, urgency) {
    const timestamp = new Date().toISOString();
    worker.postMessage({ action: 'log', data: { timestamp, event_type, message, metadata, urgency } });
    await db.logs.add({ timestamp, event_type, message, metadata, urgency });
    const compressedLog = LZString.compressToUTF16(JSON.stringify({ timestamp, event_type, message, metadata, urgency }));
    redaxios.post(`${API_BASE_URL}/log-sync`, { log: compressedLog }).catch(err => logError(`Sync Error: ${err.message}`, 'Check server', err.stack, 'MEDIUM'));
}

function logError(message, analysis, stack, urgency) {
    const timestamp = new Date().toISOString();
    const errorMessage = `[${timestamp}] ERROR: ${message}\nAnalysis: ${analysis}\nTraceback: ${stack || 'No stack'}`;
    errorLog.push(errorMessage);
    logQueue.push(`<span class="error">${errorMessage}</span>`);
    if (logQueue.length > 50) logQueue.shift();
    updateConsole();
    logEvent('error', message, { analysis, stack }, urgency);
}

function updateConsole(logs = null) {
    const consoleDiv = document.getElementById('console');
    if (!consoleDiv) return logError('Console Update Error', 'Missing console div', '', 'CRITICAL');
    consoleDiv.innerHTML = logs ? Mustache.render('{{#logs}}<p>{{message}}</p>{{/logs}}', { logs }) : logQueue.join('');
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
}

async function createVial() {
    try {
        const fileInput = document.getElementById('file-input');
        const apiInput = document.getElementById('api-input');
        const activeVials = vials.filter(v => v.status === 'running').length;
        if (activeVials >= 4) throw new Error('Maximum 4 vials');
        const vialIndex = vials.findIndex(v => v.status === 'stopped');
        if (vialIndex === -1) throw new Error('No free vials');
        const vialId = `vial_${Math.floor(100000 + Math.random() * 900000)}`;
        let code = apiInput?.value?.trim() || '';
        if (fileInput?.files?.length > 0) code = await fileInput.files[0].text();
        const vialData = { id: vialId, code: { js: code || 'console.log("Hello, Vial!");' }, training: { model: 'default', epochs: 5 } };
        const res = await redaxios.post(`${API_BASE_URL}/vial`, vialData);
        const { id, latency, createdAt, codeLength } = res.data;
        vials[vialIndex] = { name: id, latencyHistory: [latency], status: 'running', code, filePath: `/uploads/vial${id}.js`, createdAt, codeLength };
        logEvent('vial', `Created vial ${id}`, { latency, codeLength }, 'INFO');
        updateVialStatsUI();
    } catch (err) {
        logError(`Create Vial Error: ${err.message}`, 'Check backend or input', err.stack, 'CRITICAL');
    }
}

async function troubleshootVials() {
    try {
        const res = await redaxios.get(`${API_BASE_URL}/vials`);
        vials = res.data.map(vial => ({
            name: vial.id, latencyHistory: vial.latencyHistory, status: vial.status, code: vial.code.js, filePath: vial.filePath, createdAt: vial.createdAt, codeLength: vial.codeLength
        }));
        vials.forEach(vial => {
            if (vial.status === 'running') {
                const avgLatency = vial.latencyHistory.length ? (vial.latencyHistory.reduce((a, b) => a + b, 0) / vial.latencyHistory.length).toFixed(2) : '0.00';
                logEvent('troubleshoot', `Vial ${vial.name}: Status ${vial.status}, Latency ${avgLatency}ms`, { codeLength: vial.codeLength }, 'INFO');
            }
        });
        updateVialStatsUI();
    } catch (err) {
        logError(`Troubleshoot Error: ${err.message}`, 'Check backend', err.stack, 'HIGH');
    }
}

async function toggleMonitor() {
    try {
        monitorActive = !monitorActive;
        const monitorBtn = document.getElementById('monitor-btn');
        monitorBtn.classList.toggle('active-monitor', monitorActive);
        logEvent('monitor', monitorActive ? 'Monitoring started' : 'Monitoring stopped', {}, 'INFO');
        if (monitorActive) {
            monitorInterval = setInterval(async () => {
                const res = await redaxios.get(`${API_BASE_URL}/vials`);
                vials = res.data.map(vial => ({
                    name: vial.id, latencyHistory: vial.latencyHistory, status: vial.status, code: vial.code.js, filePath: vial.filePath, createdAt: vial.createdAt, codeLength: vial.codeLength
                }));
                updateVialStatsUI();
            }, 5000);
        } else {
            clearInterval(monitorInterval);
        }
    } catch (err) {
        logError(`Monitor Error: ${err.message}`, 'Check backend or DOM', err.stack, 'HIGH');
    }
}

async function trainVials() {
    try {
        trainActive = !trainActive;
        const trainBtn = document.getElementById('train-btn');
        trainBtn.classList.toggle('active-train', trainActive);
        logEvent('training', trainActive ? 'Training started' : 'Training stopped', {}, 'INFO');
        if (trainActive) {
            const inputData = document.getElementById('api-input')?.value?.trim() || '';
            if (!inputData) throw new Error('No input data');
            for (const vial of vials) {
                if (vial.status === 'running') {
                    const res = await redaxios.post(`${API_BASE_URL}/train`, { id: vial.name, input: inputData });
                    vial.latencyHistory.push(res.data.latency);
                    vial.code = inputData;
                    vial.codeLength = res.data.codeLength;
                    logEvent('training', `Trained vial ${vial.name}`, { latency: res.data.latency }, 'INFO');
                }
            }
            trainTimeout = setTimeout(() => { trainActive = false; trainBtn.classList.remove('active-train'); }, 3000);
        } else {
            clearTimeout(trainTimeout);
        }
    } catch (err) {
        logError(`Train Error: ${err.message}`, 'Check backend or input', err.stack, 'CRITICAL');
    }
}

async function voidVials() {
    try {
        await redaxios.post(`${API_BASE_URL}/destroy`);
        vials = vials.map(() => ({ name: '', latencyHistory: [], status: 'stopped', code: '', filePath: '', createdAt: '', codeLength: 0 }));
        logQueue = ['Vial MCP Controller initialized'];
        updateConsole();
        logEvent('void', 'All vials destroyed', {}, 'HIGH');
    } catch (err) {
        logError(`VOID Error: ${err.message}`, 'Check backend', err.stack, 'HIGH');
    }
}

async function saveVialAsMarkdown() {
    try {
        const res = await redaxios.get(`${API_BASE_URL}/vials`);
        const content = res.data.map(vial => Mustache.render(
            '# Vial Agent: {{id}}\n\nStatus: {{status}}\nCreated: {{createdAt}}\nCode:\n```js\n{{code.js}}\n```\nLatency: {{latencyHistory}} ms\n',
            vial
        )).join('---\n\n');
        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vial_export_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
        a.click();
        URL.revokeObjectURL(url);
        logEvent('export', 'Exported vials as Markdown', {}, 'INFO');
    } catch (err) {
        logError(`Export Error: ${err.message}`, 'Check backend or browser', err.stack, 'HIGH');
    }
}

function updateVialStatsUI() {
    const vialStatsDiv = document.getElementById('vial-stats');
    if (!vialStatsDiv) return logError('UI Update Error', 'Missing vial-stats div', '', 'CRITICAL');
    vialStatsDiv.innerHTML = vials.map((vial, index) => `
        <div class="progress-container">
            <span class="progress-label">Vial ${index + 1}:</span>
            <div class="progress-bar"><div class="progress-fill" id="vial${index + 1}-bar" style="width: ${vial.latencyHistory.length ? Math.min(100, vial.latencyHistory[vial.latencyHistory.length - 1] / 2) : 0}%"></div></div>
            <span id="vial${index + 1}-value">${vial.latencyHistory.length ? vial.latencyHistory[vial.latencyHistory.length - 1].toFixed(2) : 0} ms</span>
        </div>
    `).join('');
}

function setupEventListeners() {
    document.getElementById('create-vial-btn').addEventListener('click', createVial);
    document.getElementById('troubleshoot-btn').addEventListener('click', troubleshootVials);
    document.getElementById('monitor-btn').addEventListener('click', toggleMonitor);
    document.getElementById('train-btn').addEventListener('click', trainVials);
    document.getElementById('void-btn').addEventListener('click', voidVials);
    document.getElementById('export-btn').addEventListener('click', saveVialAsMarkdown);
    document.getElementById('file-input').addEventListener('change', async () => {
        const file = document.getElementById('file-input').files[0];
        if (file) logEvent('file', `Uploaded file: ${file.name}`, { size: file.size }, 'INFO');
    });
    document.getElementById('api-input').addEventListener('change', () => {
        const input = document.getElementById('api-input').value;
        if (input) logEvent('api', `API URL: ${input}`, {}, 'INFO');
    });
}

window.addEventListener('load', () => {
    logEvent('system', 'Vial MCP Controller initialized', {}, 'INFO');
    setupEventListeners();
});

// Instructions:
// - Tree-shaken bundle for vial.html
// - Dependencies: redaxios, lz-string, mustache, dexie
// - Logs to SQLite WASM and IndexedDB, syncs to server-side SQLite
// - Build: `scripts/build.sh`
// - Run: Loaded by vial.html
