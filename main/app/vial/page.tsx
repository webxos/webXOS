// main/app/vial/page.tsx
'use client';

import { useState, useEffect } from 'react';
import styles from './vial.module.css';
import { login } from '../../server/mcp/functions/auth';
import { listResources, callTool } from '../../server/mcp/functions/mcp';
import { getWalletBalance, sendTransaction } from '../../server/mcp/functions/wallet';

export default function Vial() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('apiKey'));
  const [isOffline, setIsOffline] = useState(true);
  const [logQueue, setLogQueue] = useState(['Vial MCP Controller initialized. Use /help for API commands.']);
  const [walletBalance, setWalletBalance] = useState(0);
  const [reputation, setReputation] = useState(0);
  const [vials, setVials] = useState([
    { id: 'vial1', status: 'stopped', latency: 0, tasks: [], isTraining: false },
    { id: 'vial2', status: 'stopped', latency: 0, tasks: [], isTraining: false },
    { id: 'vial3', status: 'stopped', latency: 0, tasks: [], isTraining: false },
    { id: 'vial4', status: 'stopped', latency: 0, tasks: [], isTraining: false },
  ]);
  const [errorMessage, setErrorMessage] = useState('');
  const [apiCredentials, setApiCredentials] = useState({ key: null, secret: null });
  const [showApiPopup, setShowApiPopup] = useState(false);
  const [command, setCommand] = useState('');

  useEffect(() => {
    if (isAuthenticated && !isOffline) {
      fetchWalletBalance();
      updateVialStatus();
    }
  }, [isAuthenticated, isOffline]);

  const fetchWalletBalance = async () => {
    try {
      const userId = localStorage.getItem('userId');
      const balanceData = await getWalletBalance(userId);
      setWalletBalance(balanceData.balance);
      setReputation(balanceData.reputation || 0);
      updateLogQueue(`$WEBXOS Balance: ${balanceData.balance.toFixed(4)} | Reputation: ${balanceData.reputation || 0}`);
    } catch (error) {
      showError(`Failed to fetch wallet balance: ${error.message}`);
    }
  };

  const updateVialStatus = () => {
    setVials((prev) =>
      prev.map((vial) => ({
        ...vial,
        latency: isOffline ? 0 : Math.floor(Math.random() * (200 - 50 + 1)) + 50,
        status: vial.isTraining ? 'running' : vial.status,
      }))
    );
  };

  const handleAuthenticate = async () => {
    try {
      const isOnline = confirm('Authenticate in online mode? Cancel for offline mode.');
      setIsOffline(!isOnline);
      const username = prompt('Enter username:');
      const password = prompt('Enter password:');
      if (username && password) {
        await login(username, password);
        setIsAuthenticated(true);
        updateLogQueue(`Authentication successful (${isOnline ? 'online' : 'offline'} mode).`);
        if (isOnline) {
          await fetchWalletBalance();
          setApiCredentials({ key: 'temp-key', secret: 'temp-secret' }); // Placeholder
        }
        updateVialStatus();
      }
    } catch (error) {
      showError(`Authentication failed: ${error.message}`);
    }
  };

  const handleVoid = () => {
    setIsAuthenticated(false);
    setIsOffline(true);
    setWalletBalance(0);
    setReputation(0);
    setApiCredentials({ key: null, secret: null });
    setVials(vials.map((vial) => ({ ...vial, status: 'stopped', tasks: [], isTraining: false, latency: 0 })));
    setLogQueue(['Vial MCP Controller initialized. Use /help for API commands.']);
    updateLogQueue('All data voided');
  };

  const handleTroubleshoot = () => {
    updateLogQueue(`Troubleshoot: System in ${isOffline ? 'offline' : 'online'} mode.`);
  };

  const handleQuantumLink = async () => {
    if (!isAuthenticated) {
      showError('Not authenticated: Please authenticate first');
      return;
    }
    try {
      setVials((prev) => prev.map((vial) => ({ ...vial, isTraining: true, status: 'running' })));
      await callTool('simulate_quantum_circuit', { num_qubits: 2, gates: ['H', 'CNOT'] });
      updateLogQueue('Quantum link activated. Agents synced.');
      setTimeout(() => {
        setVials((prev) => prev.map((vial) => ({ ...vial, isTraining: false })));
      }, 1000);
    } catch (error) {
      showError(`Quantum link failed: ${error.message}`);
      setVials((prev) => prev.map((vial) => ({ ...vial, isTraining: false })));
    }
  };

  const handleCommand = async (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!command.trim()) return;
      if (!isAuthenticated) {
        showError('Not authenticated: Please authenticate first');
        return;
      }
      const parts = command.trim().split(' ');
      const cmd = parts[0].toLowerCase();
      if (cmd === '/help') {
        updateLogQueue(`Available commands:
- /prompt <vial> <text>: Send prompt to vial
- /task <vial> <task>: Assign task to vial
- /config <vial> <key> <value>: Set vial config
- /status: Show vial statuses`);
      } else if (cmd === '/prompt') {
        if (parts.length < 3) {
          showError('Invalid command: Usage: /prompt <vial> <text>');
          return;
        }
        const vialId = parts[1];
        const promptText = parts.slice(2).join(' ');
        if (promptText.match(/(system|admin|root|eval|exec)/i)) {
          showError('Prompt contains restricted keywords');
          return;
        }
        try {
          await callTool('create_note', { title: `Prompt for ${vialId}`, content: promptText });
          updateLogQueue(`Prompt sent to ${vialId}: ${promptText}`);
          setVials((prev) =>
            prev.map((v) => (v.id === vialId ? { ...v, status: 'running' } : v))
          );
        } catch (error) {
          showError(`Failed to send prompt: ${error.message}`);
        }
      } else if (cmd === '/status') {
        const status = vials.map((v) => `${v.id}: ${v.status}, Tasks: ${v.tasks.join(', ') || 'none'}`).join('\n');
        updateLogQueue(`Vial Status:\n${status}`);
      } else {
        showError(`Unknown command: ${cmd}. Use /help for available commands.`);
      }
      setCommand('');
    }
  };

  const showError = (message) => {
    setErrorMessage(message);
    setTimeout(() => setErrorMessage(''), 5000);
  };

  const updateLogQueue = (message) => {
    setLogQueue((prev) => {
      const newQueue = [...prev.filter((log) => !log.includes('$WEBXOS Balance')), message];
      if (isAuthenticated) {
        const balanceMessage = `$WEBXOS Balance: ${isOffline ? 'N/A (Offline)' : walletBalance.toFixed(4)} | Reputation: ${isOffline ? 'N/A (Offline)' : reputation}`;
        newQueue.push(balanceMessage);
      }
      return newQueue.slice(-50);
    });
  };

  return (
    <div className={`${styles.container} ${vials.some((v) => v.isTraining) ? styles.trainGlow : ''}`}>
      <h1>Vial MCP Controller</h1>
      <div id="console" className={`${styles.console} ${vials.some((v) => v.isTraining) ? styles.activeTrain : ''}`}>
        {logQueue.map((log, i) => (
          <p key={i} className={log.includes('Balance') ? styles.balance : log.includes('error') ? styles.error : styles.command}>
            {log}
          </p>
        ))}
      </div>
      {errorMessage && <div id="error-notification" className={`${styles.errorNotification} ${styles.visible}`}>{errorMessage}</div>}
      {showApiPopup && (
        <div id="api-popup" className={`${styles.apiPopup} ${styles.visible}`}>
          <h2>API Access Credentials</h2>
          <textarea
            id="api-input"
            readOnly
            value={apiCredentials.key ? `API Key: ${apiCredentials.key}\nAPI Secret: ${apiCredentials.secret}` : ''}
          />
          <button onClick={() => setApiCredentials({ key: 'new-key', secret: 'new-secret' })}>Generate New Credentials</button>
          <button onClick={() => setShowApiPopup(false)}>Close</button>
        </div>
      )}
      <div className={styles.buttonGroup}>
        <button
          className={`${styles.button} ${isAuthenticated ? styles.activeMonitor : ''}`}
          onClick={handleAuthenticate}
        >
          Authenticate
        </button>
        <button className={styles.button} onClick={handleVoid}>
          Void
        </button>
        <button className={styles.button} onClick={handleTroubleshoot}>
          Troubleshoot
        </button>
        <button
          className={`${styles.button} ${vials.some((v) => v.isTraining) ? styles.activeTrain : ''}`}
          onClick={handleQuantumLink}
          disabled={!isAuthenticated}
        >
          Quantum Link
        </button>
        <button className={styles.button} disabled>
          Export
        </button>
        <button className={styles.button} disabled>
          Import
        </button>
        <button
          className={styles.button}
          onClick={() => setShowApiPopup(true)}
          disabled={!isAuthenticated || isOffline}
        >
          API Access
        </button>
      </div>
      <textarea
        id="prompt-input"
        className={styles.promptInput}
        placeholder="Enter git commands for API (e.g., /prompt vial1 train dataset)..."
        value={command}
        onChange={(e) => setCommand(e.target.value)}
        onKeyDown={handleCommand}
      />
      <div id="vial-status-bars" className={styles.vialStatusBars}>
        {vials.map((vial) => (
          <div key={vial.id} className={styles.progressContainer}>
            <span className={styles.progressLabel}>{vial.id}</span>
            <div className={styles.progressBar}>
              <div
                className={`${styles.progressFill} ${vial.status === 'stopped' ? styles.offlineGrey : vial.isTraining ? styles.training : ''}`}
                style={{ width: vial.status === 'running' ? '100%' : '0%' }}
              />
            </div>
            <span
              className={`${styles.statusText} ${vial.isTraining ? styles.training : isOffline ? styles.offlineGrey : styles.online}`}
            >
              Latency: {vial.latency}ms | Mode: {vial.isTraining ? 'Training' : isOffline ? 'Offline' : 'Online'}
            </span>
          </div>
        ))}
      </div>
      <footer className={styles.footer}>WebXOS Vial MCP Controller | {isOffline ? 'Offline' : 'Online'} Mode | 2025 | v2.7</footer>
    </div>
  );
}
