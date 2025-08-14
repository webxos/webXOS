// main/app/vial/page.tsx
'use client';

import { useState, useEffect, useCallback, useActionState, useOptimistic } from 'react';
import { useAccount, useBalance, useWriteContract } from 'wagmi';
import { parseEther } from 'viem';
import { trace } from '@opentelemetry/api';
import { useRouter } from 'next/navigation';
import { vialAbi } from '../../server/mcp/wallet/vial_abi';
import styles from './vial.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';
const GEMINI_API_KEY = process.env.NEXT_PUBLIC_GEMINI_API_KEY || 'YOUR_GEMINI_API_KEY';
const OCI_COMPARTMENT_ID = process.env.NEXT_PUBLIC_OCI_COMPARTMENT_ID || 'YOUR_OCI_COMPARTMENT_ID';
const VERSION = '2.8';

interface Vial {
  id: string;
  status: 'stopped' | 'running';
  code: string;
  codeLength: number;
  wallet: { address: string | null; balance: number; hash: string | null; webxos: number; transactions: any[] };
  tasks: string[];
  latency: number;
}

interface AIResponse {
  response: string;
  provider: string;
}

const VialMCPController = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isOffline, setIsOffline] = useState(true);
  const [apiKey, setApiKey] = useState<string | null>(typeof window !== 'undefined' ? localStorage.getItem('apiKey') : null);
  const [userId, setUserId] = useState<string | null>(typeof window !== 'undefined' ? localStorage.getItem('userId') : null);
  const { address, isConnected } = useAccount();
  const { data: balance } = useBalance({ address });
  const { writeContract, isPending } = useWriteContract();
  const [optimisticWallet, setOptimisticWallet] = useOptimistic({ address: null, balance: 0, hash: null, webxos: 0.0, transactions: [] }, (state, newWallet) => newWallet);
  const [reputation, setReputation] = useState(0);
  const [vials, setVials] = useState<Vial[]>(Array(4).fill().map((_, i) => ({
    id: `vial${i + 1}`,
    status: 'stopped',
    code: '',
    codeLength: 0,
    wallet: { address: null, balance: 0, hash: null, webxos: 0.0, transactions: [] },
    tasks: [],
    latency: 0
  })));
  const [logQueue, setLogQueue] = useState<string[]>([
    `<p>Vial MCP Controller initialized. Use /help for commands. Authenticate to enable all features.</p>`,
    `<p class="balance">$WEBXOS Balance: ${optimisticWallet.webxos.toFixed(4)} | Reputation: 0</p>`
  ]);
  const [errorMessage, setErrorMessage] = useState('');
  const [isApiPopupVisible, setIsApiPopupVisible] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const router = useRouter();

  const tracer = trace.getTracer('vial-mcp-controller');

  const logEvent = useCallback((eventType: string, message: string, metadata = {}) => {
    const timestamp = new Date().toISOString();
    const logId = Date.now();
    const formattedMessage = `<p class="${eventType === 'error' ? 'error' : 'command'}">[${timestamp}] ${message} [ID:${logId}]</p>`;
    setLogQueue(prev => [...prev, formattedMessage].slice(-50));
    if (eventType === 'error') {
      setErrorMessage(message);
      fetch(`${API_BASE}/log_error`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ error: message, stack: metadata.error || 'No stack trace', endpoint: metadata.endpoint || 'unknown', timestamp, source: 'frontend', rawResponse: metadata.rawResponse || '' })
      }).catch(() => localStorage.setItem(`error_${logId}`, JSON.stringify({ error: message, stack: metadata.error || 'No stack trace', endpoint: metadata.endpoint || 'unknown', timestamp })));
    }
  }, [apiKey]);

  const fetchWithRetry = useCallback(async (url: string, options: RequestInit, retries = 3, baseDelay = 1000): Promise<{ ok: boolean; json: () => Promise<any> }> => {
    const span = tracer.startSpan('fetch_with_retry', { attributes: { url, retries } });
    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, { ...options, headers: { ...options.headers, 'X-Gemini-API-Key': GEMINI_API_KEY, 'OCI-Compartment-Id': OCI_COMPARTMENT_ID } });
        span.setStatus({ code: 1 });
        span.end();
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        return { ok: response.ok, json: () => response.json() };
      } catch (error: any) {
        if (i === retries - 1) {
          span.recordException(error);
          span.setStatus({ code: 2 });
          span.end();
          logEvent('error', `Fetch failed: ${error.message}`, { error: error.stack, endpoint: url, rawResponse: error.message });
          setIsOffline(true);
          return { ok: false, json: () => ({ error: 'Offline mode' }) };
        }
        await new Promise(resolve => setTimeout(resolve, baseDelay * Math.pow(2, i)));
      }
    }
    span.end();
    return { ok: false, json: () => ({ error: 'Fetch failed' }) };
  }, [logEvent, tracer]);

  const checkBackendStatus = useCallback(async () => {
    const span = tracer.startSpan('check_backend_status');
    try {
      const response = await fetchWithRetry(`${API_BASE}/health`, { method: 'GET' });
      const data = await response.json();
      setIsOffline(data.status !== 'healthy');
      logEvent('system', data.status === 'healthy' ? `Backend online at ${API_BASE}` : `Backend offline at ${API_BASE}`, { endpoint: `${API_BASE}/health` });
      span.setStatus({ code: 1 });
      return data;
    } catch (error: any) {
      setIsOffline(true);
      logEvent('error', `Backend offline: ${error.message}`, { error: error.stack, endpoint: `${API_BASE}/health`, rawResponse: error.message });
      span.setStatus({ code: 2 });
      span.end();
      return { status: 'unhealthy', mongo: false, services: [] };
    }
  }, [fetchWithRetry, logEvent, tracer]);

  const [authState, authAction] = useActionState(async (_: any, formData: FormData) => {
    const userIdInput = formData.get('userId')?.toString();
    if (!userIdInput) {
      setErrorMessage('User ID required');
      return { error: 'User ID required' };
    }
    try {
      const backendStatus = await checkBackendStatus();
      setIsOffline(backendStatus.status !== 'healthy');
      if (backendStatus.status !== 'healthy') {
        setApiKey(`OFFLINE-${crypto.randomUUID()}`);
        setUserId(userIdInput);
        setOptimisticWallet({ address: 'mock-wallet', balance: 0, hash: 'mock-hash', webxos: 0.0, transactions: [] });
        setIsAuthenticated(true);
        logEvent('system', `Authenticated offline with API key: ${apiKey}`);
        return { success: true, apiKey: `OFFLINE-${crypto.randomUUID()}`, userId: userIdInput };
      }
      const response = await fetchWithRetry(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: userIdInput })
      });
      const data = await response.json();
      setApiKey(data.apiKey);
      setUserId(data.userId);
      setOptimisticWallet({ address: data.walletAddress, balance: 0, hash: data.walletHash, webxos: 0.0, transactions: [] });
      localStorage.setItem('apiKey', data.apiKey);
      localStorage.setItem('userId', data.userId);
      setIsAuthenticated(true);
      logEvent('system', `Authenticated online with API key: ${data.apiKey}`);
      await fetchWithRetry(`${API_BASE}/wallet/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${data.apiKey}` },
        body: JSON.stringify({ userId: data.userId, ...optimisticWallet })
      });
      return { success: true, apiKey: data.apiKey, userId: data.userId };
    } catch (error: any) {
      setIsOffline(true);
      setApiKey(`OFFLINE-${crypto.randomUUID()}`);
      setUserId(userIdInput);
      setOptimisticWallet({ address: 'mock-wallet', balance: 0, hash: 'mock-hash', webxos: 0.0, transactions: [] });
      setIsAuthenticated(true);
      logEvent('error', `Authentication failed: ${error.message}`, { error: error.stack, endpoint: `${API_BASE}/auth/login`, rawResponse: error.message });
      setErrorMessage(`Authentication failed: ${error.message}. Limited functionality available.`);
      return { error: error.message };
    }
  }, {});

  const handleGitCommand = useCallback(async (command: string) => {
    const span = tracer.startSpan('handle_git_command', { attributes: { command } });
    const sanitizedCommand = command.trim().replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '').replace(/[<>{}]/g, '');
    const parts = sanitizedCommand.split(' ');
    const cmd = parts[0].toLowerCase();
    try {
      if (cmd === '/help') {
        logEvent('command', `Available commands:
- /help: Show this help
- /status: Show vial statuses
- /troubleshoot: Run diagnostics
- /void: Reset all vials
- /prompt <vial> <text>: Send prompt to vial
- /task <vial> <task>: Assign task to vial
- /config <vial> <key> <value>: Set vial config
- /auth <userId>: Authenticate with user ID
- /wallet create: Create a wallet
- /wallet import: Import a wallet
- /monitor: Toggle vial monitoring
- /train: Toggle vial training
- /reset: Reset all vials`);
      } else if (cmd === '/status') {
        const status = vials.map(v => `${v.id}: ${v.status}, Tasks: ${v.tasks.join(', ') || 'none'}`).join('\n');
        logEvent('command', `Vial Status:\n${status}`);
      } else if (cmd === '/troubleshoot') {
        await checkBackendStatus();
        logEvent('command', `Diagnostics: Authenticated: ${isAuthenticated}, Mode: ${isOffline ? 'Offline' : 'Online'}, API Key: ${apiKey || 'None'}`);
      } else if (cmd === '/void') {
        if (!isOffline) await fetchWithRetry(`${API_BASE}/vials/void`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${apiKey}` } });
        setVials(vials.map(v => ({ ...v, status: 'stopped', tasks: [], code: '', codeLength: 0 })));
        logEvent('command', 'All vials reset');
      } else if (!isAuthenticated) {
        throw new Error('Authenticate first to use this command');
      } else if (cmd === '/prompt') {
        if (parts.length < 3) throw new Error('Usage: /prompt <vial> <text>');
        const vialId = parts[1];
        const promptText = parts.slice(2).join(' ');
        if (!vials.find(v => v.id === vialId)) throw new Error(`Invalid vial: ${vialId}`);
        if (isOffline) throw new Error('Cannot send prompts offline');
        const response = await fetchWithRetry(`${API_BASE}/vials/${vialId}/prompt`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({ vialId, prompt: promptText })
        });
        const data = await response.json();
        logEvent('command', `Prompt sent to ${vialId}: ${data.response}`);
      } else if (cmd === '/task') {
        if (parts.length < 3) throw new Error('Usage: /task <vial> <task>');
        const vialId = parts[1];
        const task = parts.slice(2).join(' ');
        if (!vials.find(v => v.id === vialId)) throw new Error(`Invalid vial: ${vialId}`);
        if (isOffline) throw new Error('Cannot assign tasks offline');
        const response = await fetchWithRetry(`${API_BASE}/vials/${vialId}/task`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({ vialId, task })
        });
        const data = await response.json();
        setVials(vials.map(v => v.id === vialId ? { ...v, tasks: [...v.tasks, task], status: 'running' } : v));
        logEvent('command', `Task assigned to ${vialId}: ${data.status}`);
      } else if (cmd === '/config') {
        if (parts.length < 4) throw new Error('Usage: /config <vial> <key> <value>');
        const vialId = parts[1];
        const key = parts[2];
        const value = parts.slice(3).join(' ');
        if (!vials.find(v => v.id === vialId)) throw new Error(`Invalid vial: ${vialId}`);
        if (isOffline) throw new Error('Cannot set config offline');
        const response = await fetchWithRetry(`${API_BASE}/vials/${vialId}/config`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({ vialId, key, value })
        });
        const data = await response.json();
        setVials(vials.map(v => v.id === vialId ? { ...v, status: 'running' } : v));
        logEvent('command', `Config set for ${vialId}: ${key}=${value}`);
      } else if (cmd === '/auth') {
        if (parts.length < 2) throw new Error('Usage: /auth <userId>');
        setUserId(parts[1]);
        authAction(new FormData(Object.entries({ userId: parts[1] })));
      } else if (cmd === '/wallet') {
        if (isOffline) throw new Error('Cannot manage wallet offline');
        if (parts[1] === 'create') {
          const response = await fetchWithRetry(`${API_BASE}/wallet/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
            body: JSON.stringify({ userId, address: address || 'new-wallet', balance: 0, hash: 'new-hash', webxos: 0.0 })
          });
          const data = await response.json();
          setOptimisticWallet({ address: data.address, balance: 0, hash: data.hash, webxos: 0.0, transactions: [] });
          logEvent('command', data.status);
        } else if (parts[1] === 'import') {
          document.getElementById('file-input')?.click();
        } else {
          throw new Error('Usage: /wallet create or /wallet import');
        }
      } else if (cmd === '/monitor') {
        setIsMonitoring(!isMonitoring);
        logEvent('command', `Monitoring ${!isMonitoring ? 'started' : 'stopped'}`);
      } else if (cmd === '/train') {
        setIsTraining(!isTraining);
        logEvent('command', `Training ${!isTraining ? 'started' : 'stopped'}`);
      } else if (cmd === '/reset') {
        if (!isOffline) await fetchWithRetry(`${API_BASE}/vials/void`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${apiKey}` } });
        setVials(vials.map(v => ({ ...v, status: 'stopped', tasks: [], code: '', codeLength: 0 })));
        logEvent('command', 'All vials reset');
      } else {
        throw new Error(`Unknown command: ${cmd}. Use /help`);
      }
      span.setStatus({ code: 1 });
    } catch (error: any) {
      logEvent('error', error.message, { error: error.stack });
      setErrorMessage(error.message);
      span.setStatus({ code: 2 });
    } finally {
      span.end();
    }
  }, [vials, isAuthenticated, isOffline, apiKey, userId, isMonitoring, isTraining, fetchWithRetry, logEvent, authAction, address, tracer]);

  const importWallet = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const span = tracer.startSpan('import_wallet');
    try {
      if (!isAuthenticated || isOffline) throw new Error('Cannot import wallet offline or without authentication');
      const file = event.target.files?.[0];
      if (!file || !file.name.endsWith('.md')) throw new Error('Invalid file: Use .md');
      const text = await file.text();
      if (!text.includes('## Wallet')) throw new Error('Invalid Markdown format');
      setOptimisticWallet({ address: 'importing', balance: 0, hash: 'importing', webxos: 0.0, transactions: [] });
      const response = await fetchWithRetry(`${API_BASE}/wallet/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ userId, walletData: text })
      });
      const data = await response.json();
      setOptimisticWallet({ address: data.address, balance: data.balance, hash: data.hash, webxos: data.webxos, transactions: data.transactions });
      logEvent('command', `Imported wallet from ${file.name}`);
      event.target.value = '';
      span.setStatus({ code: 1 });
    } catch (error: any) {
      logEvent('error', `Import failed: ${error.message}`, { error: error.stack, endpoint: `${API_BASE}/wallet/import` });
      setErrorMessage(`Import failed: ${error.message}`);
      span.setStatus({ code: 2 });
    } finally {
      span.end();
    }
  }, [isAuthenticated, isOffline, apiKey, userId, fetchWithRetry, logEvent, setOptimisticWallet, tracer]);

  const exportWallet = useCallback(async () => {
    const span = tracer.startSpan('export_wallet');
    try {
      if (!isAuthenticated || isOffline) throw new Error('Cannot export wallet offline or without authentication');
      const data = {
        markdown: `# WebXOS Vial Wallet Export\n\n## Wallet\n- Address: ${optimisticWallet.address || 'none'}\n- Balance: ${optimisticWallet.webxos.toFixed(4)} $WEBXOS\n- Hash: ${optimisticWallet.hash || 'none'}\n- Transactions: ${JSON.stringify(optimisticWallet.transactions, null, 2).replace(/\n/g, '\n  ')}\n`
      };
      const blob = new Blob([data.markdown], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vial_wallet_export_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
      a.click();
      URL.revokeObjectURL(url);
      logEvent('command', 'Wallet exported');
      span.setStatus({ code: 1 });
    } catch (error: any) {
      logEvent('error', `Export failed: ${error.message}`, { error: error.stack });
      setErrorMessage(`Export failed: ${error.message}`);
      span.setStatus({ code: 2 });
    } finally {
      span.end();
    }
  }, [isAuthenticated, isOffline, optimisticWallet, logEvent, tracer]);

  useEffect(() => {
    const initialize = async () => {
      await checkBackendStatus();
      if (apiKey && userId && !apiKey.startsWith('OFFLINE-')) {
        authAction(new FormData(Object.entries({ userId })));
      }
    };
    initialize();
    const interval = setInterval(() => {
      if (isAuthenticated && !isOffline) checkBackendStatus();
    }, 15000);
    return () => clearInterval(interval);
  }, [checkBackendStatus, authAction, isAuthenticated, isOffline, apiKey, userId]);

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(''), 10000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  return (
    <div className="flex flex-col items-center h-screen">
      <h1 className="text-2xl font-bold text-green-500">Vial MCP Controller</h1>
      <div className={`${styles.console} ${isMonitoring ? styles.activeMonitor : ''} ${isTraining ? styles.activeTrain : ''}`} dangerouslySetInnerHTML={{ __html: logQueue.join('') }} />
      <div className={`${styles.errorNotification} ${errorMessage ? styles.visible : ''} ${errorMessage.includes('success') ? styles.success : ''}`}>
        {errorMessage}
      </div>
      <div className={`${styles.apiPopup} ${isApiPopupVisible ? styles.visible : ''}`}>
        <h2>API Access Credentials</h2>
        <textarea readOnly value={`Key: ${apiKey || 'None'}`} className={styles.apiInput} />
        <button
          onClick={() => fetchWithRetry(`${API_BASE}/auth/api-key/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
            body: JSON.stringify({ userId })
          }).then(res => res.json()).then(data => {
            setApiKey(data.apiKey);
            localStorage.setItem('apiKey', data.apiKey);
            logEvent('command', 'New API credentials generated');
          })}
          className={styles.apiButton}
        >
          Generate New Credentials
        </button>
        <button onClick={() => setIsApiPopupVisible(false)} className={styles.apiButton}>Close</button>
      </div>
      <div className={styles.buttonGroup}>
        <form action={authAction}>
          <input name="userId" placeholder="Enter User ID" className={styles.input} required />
          <button type="submit" className={`${styles.button} ${isAuthenticated ? styles.activeMonitor : ''}`} disabled={authState.isPending}>
            {authState.isPending ? 'Authenticating...' : 'Authenticate'}
          </button>
        </form>
        <button onClick={() => handleGitCommand('/void')} className={styles.button}>Void</button>
        <button onClick={() => handleGitCommand('/troubleshoot')} className={styles.button}>Troubleshoot</button>
        <button onClick={() => handleGitCommand('/monitor')} disabled={isOffline} className={`${styles.button} ${isMonitoring ? styles.activeMonitor : ''}`}>
          Quantum Link
        </button>
        <button onClick={exportWallet} disabled={isOffline || !isAuthenticated} className={styles.button}>Export</button>
        <button onClick={() => document.getElementById('file-input')?.click()} disabled={isOffline || !isAuthenticated} className={styles.button}>
          Import
        </button>
        <button onClick={() => setIsApiPopupVisible(true)} disabled={isOffline || !isAuthenticated} className={styles.button}>API Access</button>
      </div>
      <textarea
        id="prompt-input"
        placeholder="Enter commands (e.g., /help, /status, /prompt vial1 train dataset, /auth user123)"
        className={styles.input}
        onKeyPress={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGitCommand(e.currentTarget.value);
            e.currentTarget.value = '';
          }
        }}
      />
      <div className={styles.statusBars}>
        {vials.map(vial => (
          <div key={vial.id} className={styles.progressContainer}>
            <span className={styles.progressLabel}>{vial.id}</span>
            <div className={styles.progressBar}>
              <div className={`${styles.progressFill} ${vial.status === 'running' ? '' : styles.offlineGrey}`} style={{ width: vial.status === 'running' ? '100%' : '0%' }}></div>
            </div>
            <span className={`${styles.statusText} ${isOffline ? styles.offlineGrey : styles.online}`}>
              Latency: {vial.latency}ms | Size: {vial.codeLength} bytes | Mode: {isOffline ? 'Offline' : 'Online'}
            </span>
          </div>
        ))}
      </div>
      <footer className={styles.footer}>WebXOS Vial MCP Controller | {isOffline ? 'Offline' : 'Online'} Mode | 2025 | v{VERSION}</footer>
      <input type="file" id="file-input" accept=".md" className="hidden" onChange={importWallet} />
    </div>
  );
};

export default VialMCPController;
