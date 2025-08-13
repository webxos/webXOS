import { useState, useEffect } from 'react';
import Head from 'next/head';
import axios from 'axios';
import Fuse from 'fuse.js';
import '/public/style.css';
import '/public/neurots.js';

export default function Chatbot() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [wallet, setWallet] = useState({ transactions: [], webxos: 0.0 });
  const [apiKey, setApiKey] = useState(null);
  const [gitCommand, setGitCommand] = useState('');
  const [showHelp, setShowHelp] = useState(false);

  const sources = ['postgres', 'milvus', 'weaviate', 'pgvector', 'faiss'];
  const llms = ['llama3.3', 'mistral', 'gemma2', 'qwen', 'phi'];
  const gitCommands = [
    { command: 'git clone', description: 'Clone the repository for training data' },
    { command: 'git commit', description: 'Commit changes to training configurations' },
    { command: 'git push', description: 'Push changes to the remote repository' },
    { command: 'git pull', description: 'Pull latest updates from the repository' },
    { command: 'git branch', description: 'Create or list branches for training variants' },
    { command: 'git merge', description: 'Merge branches to consolidate training changes' }
  ];

  useEffect(() => {
    const fuse = new Fuse([], { keys: ['data'] });
    if (typeof window.initNeurots === 'function') {
      window.initNeurots();
    }
  }, []);

  const handleGenerateApiKey = async () => {
    try {
      setError(null);
      const response = await axios.post('/v1/api/generate_api_key', { user_id: 'user123' });
      setApiKey(response.data.api_key);
    } catch (err) {
      setError(`API key generation failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleRetrieve = async (source) => {
    try {
      setError(null);
      const response = await axios.post('/api/retrieve', { user_id: 'user123', query, source, wallet, format: 'json' }, {
        headers: { Authorization: `Bearer ${apiKey}` }
      });
      setResults(response.data.data);
      setWallet(response.data.wallet || wallet);
    } catch (err) {
      setError(`Retrieval failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleLLM = async (model) => {
    try {
      setError(null);
      const response = await axios.post('/v1/api/llm', { user_id: 'user123', prompt: query, model, wallet, format: 'json' }, {
        headers: { Authorization: `Bearer ${apiKey}` }
      });
      setResults([response.data.response]);
      setWallet(response.data.wallet || wallet);
    } catch (err) {
      setError(`LLM call failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleGitCommand = async () => {
    try {
      setError(null);
      const response = await axios.post('/v1/api/git', { user_id: 'user123', command: gitCommand, repo_url: 'https://github.com/webxos/webxos.git', wallet }, {
        headers: { Authorization: `Bearer ${apiKey}` }
      });
      setResults([response.data.output]);
      setWallet(response.data.wallet || wallet);
    } catch (err) {
      setError(`Git command failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  return (
    <div className="container">
      <Head>
        <title>Chatbot UI</title>
        <link rel="stylesheet" href="/style.css" />
        <script src="/neurots.js" />
        <script src="/fuse.min.js" />
      </Head>
      <h1>Chatbot UI</h1>
      {error && <div className="error">{error}</div>}
      <button onClick={handleGenerateApiKey}>Generate API Key</button>
      {apiKey && <div>API Key: {apiKey}</div>}
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter query for retrieval or LLM"
      />
      <div>
        {sources.map((source) => (
          <button key={source} onClick={() => handleRetrieve(source)}>
            Retrieve from {source}
          </button>
        ))}
      </div>
      <div>
        {llms.map((model) => (
          <button key={model} onClick={() => handleLLM(model)}>
            Query {model}
          </button>
        ))}
      </div>
      <div>
        <h2>Git Commands</h2>
        <input
          type="text"
          value={gitCommand}
          onChange={(e) => setGitCommand(e.target.value)}
          placeholder="Enter Git command (e.g., git clone)"
        />
        <button onClick={handleGitCommand}>Execute Git Command</button>
        <button onClick={() => setShowHelp(!showHelp)}>
          {showHelp ? 'Hide Help' : 'Show Help'}
        </button>
        {showHelp && (
          <div>
            <h3>Git Command Help</h3>
            <ul>
              {gitCommands.map((cmd) => (
                <li key={cmd.command}>
                  <strong>{cmd.command}</strong>: {cmd.description}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div>
        <h2>Results</h2>
        <pre>{JSON.stringify(results, null, 2)}</pre>
      </div>
      <div>
        <h2>Wallet</h2>
        <pre>{JSON.stringify(wallet, null, 2)}</pre>
      </div>
      <div id="neurots-visualization"></div>
    </div>
  );
}
