import React, { useState, useEffect } from 'https://cdn.jsdelivr.net/npm/react@18.2.0/+esm';
import ReactDOM from 'https://cdn.jsdelivr.net/npm/react-dom@18.2.0/+esm';
import axios from 'https://cdn.jsdelivr.net/npm/axios@1.7.7/+esm';
import 'https://cdn.jsdelivr.net/npm/tailwindcss@3.4.13/dist/tailwind.min.css';

const ToolsPage = () => {
  const [walletId, setWalletId] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [token, setToken] = useState('');
  const [toolName, setToolName] = useState('');
  const [task, setTask] = useState('');
  const [params, setParams] = useState('');
  const [result, setResult] = useState(null);
  const [message, setMessage] = useState('');

  const login = async () => {
    try {
      const response = await axios.post('https://localhost:8000/api/auth/login', {
        wallet_id: walletId,
        api_key: apiKey
      });
      setToken(response.data.access_token);
      setMessage('Login successful');
    } catch (error) {
      setMessage(`Login failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const executeTool = async () => {
    try {
      const response = await axios.post('https://localhost:8000/api/agents/execute', {
        agent_name: toolName,
        task: task,
        params: JSON.parse(params || '{}'),
        wallet_id: walletId
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResult(response.data);
      setMessage('Tool executed successfully');
    } catch (error) {
      setMessage(`Tool execution failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  useEffect(() => {
    if (token) {
      setMessage('');
    }
  }, [token]);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Vial MCP Tools</h1>
      <div className="mb-4">
        <input
          type="text"
          placeholder="Wallet ID"
          value={walletId}
          onChange={(e) => setWalletId(e.target.value)}
          className="border p-2 mr-2"
        />
        <input
          type="text"
          placeholder="API Key"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          className="border p-2 mr-2"
        />
        <button onClick={login} className="bg-blue-500 text-white p-2 rounded">
          Login
        </button>
      </div>
      <div className="mb-4">
        <h2 className="text-xl font-semibold">Execute Tool</h2>
        <div className="mb-2">
          <label className="block">Tool Name</label>
          <select
            value={toolName}
            onChange={(e) => setToolName(e.target.value)}
            className="border p-2"
          >
            <option value="">Select Tool</option>
            <option value="jina">Jina AI</option>
            <option value="nomic">Nomic</option>
            <option value="cogni">CogniTALLMware</option>
            <option value="llmware">LLMware</option>
          </select>
        </div>
        <div className="mb-2">
          <label className="block">Task</label>
          <input
            type="text"
            placeholder="Task (e.g., translate_content)"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="border p-2 w-full"
          />
        </div>
        <div className="mb-2">
          <label className="block">Parameters (JSON)</label>
          <textarea
            placeholder='{"content":"Hello","target_lang":"es"}'
            value={params}
            onChange={(e) => setParams(e.target.value)}
            className="border p-2 w-full"
          />
        </div>
        <button onClick={executeTool} className="bg-green-500 text-white p-2 rounded">
          Execute Tool
        </button>
      </div>
      {result && (
        <div className="mb-4">
          <h2 className="text-xl font-semibold">Result</h2>
          <pre className="bg-gray-100 p-4">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
      {message && <p className="text-red-500">{message}</p>}
    </div>
  );
};

ReactDOM.render(<ToolsPage />, document.getElementById('root'));
