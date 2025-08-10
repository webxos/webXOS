```jsx
import React, { useState, useEffect } from 'react';
import Console from './components/Console';
import VialStats from './components/VialStats';
import Auth from './components/Auth';
import axios from 'axios';
import Fuse from 'fuse.js';
import './index.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [accessKey, setAccessKey] = useState('');
  const [vials, setVials] = useState(Array(4).fill().map((_, i) => ({
    id: `vial${i+1}`,
    status: 'stopped',
    code: 'import torch\nimport torch.nn as nn\n\nclass VialAgent(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 1)\n    def forward(self, x):\n        return torch.sigmoid(self.fc(x))\n\nmodel = VialAgent()',
    codeLength: 0,
    isPython: true,
    webxosHash: generateUUID(),
    wallet: { address: null, balance: 0 },
    tasks: []
  })));
  const [wallet, setWallet] = useState({ address: null, balance: 0, hashRate: 0 });
  const [logs, setLogs] = useState(['Vial MCP Controller initialized']);
  const [error, setError] = useState('');
  const [networkId, setNetworkId] = useState(null);

  useEffect(() => {
    let interval;
    if (isAuthenticated) {
      interval = setInterval(() => {
        setWallet(prev => ({
          ...prev,
          balance: prev.balance + (1 / 60), // 1 $WEBXOS per minute
          hashRate: prev.hashRate + 1 // 1 hash/second
        }));
        setVials(prev => prev.map(vial => ({
          ...vial,
          wallet: { ...vial.wallet, balance: wallet.balance / 4 }
        })));
        axios.post('/api/wallet/update', { balance: wallet.balance, hashRate: wallet.hashRate }, {
          headers: { Authorization: `Bearer ${accessKey}` }
        }).catch(err => logEvent('error', `Wallet Update Error: ${err.message}`));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isAuthenticated, wallet.balance, wallet.hashRate, accessKey]);

  const generateUUID = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  };

  const logEvent = (type, message) => {
    setLogs(prev => [...prev, `[${new Date().toISOString()}] ${type.toUpperCase()}: ${message}`].slice(-50));
  };

  const showError = (message) => {
    setError(message);
    setTimeout(() => setError(''), 5000);
  };

  const handleAuth = async (credentials) => {
    try {
      const response = await axios.post('/api/auth', credentials);
      setAccessKey(response.data.accessKey);
      setIsAuthenticated(true);
      setWallet({ address: response.data.address, balance: 0, hashRate: 0 });
      setVials(response.data.vials);
      setNetworkId(response.data.networkId);
      logEvent('auth', 'Authentication successful');
    } catch (err) {
      logEvent('error', `Auth Error: ${err.message}`);
      showError(`Auth Error: ${err.message}`);
    }
  };

  const handleVoid = async () => {
    try {
      await axios.post('/api/void', {}, { headers: { Authorization: `Bearer ${accessKey}` } });
      setIsAuthenticated(false);
      setAccessKey('');
      setWallet({ address: null, balance: 0, hashRate: 0 });
      setVials(Array(4).fill().map((_, i) => ({
        id: `vial${i+1}`,
        status: 'stopped',
        code: 'import torch\nimport torch.nn as nn\n\nclass VialAgent(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 1)\n    def forward(self, x):\n        return torch.sigmoid(self.fc(x))\n\nmodel = VialAgent()',
        codeLength: 0,
        isPython: true,
        webxosHash: generateUUID(),
        wallet: { address: null, balance: 0 },
        tasks: []
      })));
      setNetworkId(null);
      setLogs(['Vial MCP Controller initialized']);
      logEvent('void', 'System voided');
    } catch (err) {
      logEvent('error', `Void Error: ${err.message}`);
      showError(`Void Error: ${err.message}`);
    }
  };

  const handleTrain = async (file) => {
    if (!isAuthenticated) return showError('Not authenticated');
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('networkId', networkId);
      const response = await axios.post('/api/train', formData, {
        headers: { Authorization: `Bearer ${accessKey}` }
      });
      setVials(response.data.vials);
      setWallet(prev => ({ ...prev, balance: prev.balance + response.data.balance }));
      logEvent('training', 'Training completed');
    } catch (err) {
      logEvent('error', `Train Error: ${err.message}`);
      showError(`Train Error: ${err.message}`);
    }
  };

  const handleExport = async () => {
    if (!isAuthenticated) return showError('Not authenticated');
    try {
      const response = await axios.get(`/api/export?networkId=${networkId}`, {
        headers: { Authorization: `Bearer ${accessKey}` }
      });
      const blob = new Blob([response.data.markdown], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vial_export_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
      a.click();
      URL.revokeObjectURL(url);
      logEvent('export', 'Exported vials and wallet');
    } catch (err) {
      logEvent('error', `Export Error: ${err.message}`);
      showError(`Export Error: ${err.message}`);
    }
  };

  const handleUpload = async (file) => {
    if (!isAuthenticated) return showError('Not authenticated');
    try {
      const text = await file.text();
      const fuse = new Fuse([text], { keys: ['content'], threshold: 0.3 });
      const result = fuse.search('Agentic Network');
      if (!result.length) throw new Error('Invalid .md format');
      const formData = new FormData();
      formData.append('file', file);
      formData.append('networkId', networkId);
      const response = await axios.post('/api/upload', formData, {
        headers: { Authorization: `Bearer ${accessKey}` }
      });
      setVials(response.data.vials);
      setWallet(response.data.wallet);
      setNetworkId(response.data.networkId);
      logEvent('upload', 'File uploaded and parsed');
    } catch (err) {
      logEvent('error', `Upload Error: ${err.message}`);
      showError(`Upload Error: ${err.message}`);
    }
  };

  const handlePrompt = async (prompt) => {
    if (!isAuthenticated) return showError('Not authenticated');
    try {
      const response = await axios.post('/api/comms', { prompt, networkId }, {
        headers: { Authorization: `Bearer ${accessKey}` }
      });
      logEvent('comms', `Response: ${response.data.response}`);
    } catch (err) {
      logEvent('error', `Comms Error: ${err.message}`);
      showError(`Comms Error: ${err.message}`);
    }
  };

  return (
    <div className="app">
      <h1>Vial MCP Controller</h1>
      <div className={`error-notification ${error ? 'visible' : ''}`}>{error}</div>
      <Auth isAuthenticated={isAuthenticated} onAuth={handleAuth} accessKey={accessKey} />
      <Console logs={logs} isAuthenticated={isAuthenticated} onPrompt={handlePrompt} />
      <VialStats vials={vials} wallet={wallet} />
      <div className="button-group">
        <button onClick={handleVoid} disabled={!isAuthenticated}>Void</button>
        <button onClick={() => document.getElementById('file-input').click()} disabled={!isAuthenticated}>Train</button>
        <button onClick={handleExport} disabled={!isAuthenticated}>Export</button>
        <button onClick={() => document.getElementById('file-input').click()} disabled={!isAuthenticated}>Upload</button>
        <button disabled={!isAuthenticated}>API Access</button>
      </div>
      <input type="file" id="file-input" accept=".js,.py,.txt,.md" style={{ display: 'none' }} onChange={(e) => {
        if (e.target.files[0]) {
          if (e.target.files[0].name.endsWith('.md')) handleUpload(e.target.files[0]);
          else handleTrain(e.target.files[0]);
        }
      }} />
      <footer>WebXOS Vial MCP Controller | {isAuthenticated ? 'Online' : 'Offline'} Mode | 2025 | v2.1</footer>
    </div>
  );
}

export default App;
```
