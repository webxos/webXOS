import React, { useState, useEffect } from 'https://cdn.jsdelivr.net/npm/react@18.2.0/+esm';
import ReactDOM from 'https://cdn.jsdelivr.net/npm/react-dom@18.2.0/+esm';
import axios from 'https://cdn.jsdelivr.net/npm/axios@1.7.7/+esm';
import 'https://cdn.jsdelivr.net/npm/tailwindcss@3.4.13/dist/tailwind.min.css';

const DashboardPage = () => {
  const [walletId, setWalletId] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [token, setToken] = useState('');
  const [metrics, setMetrics] = useState([]);
  const [health, setHealth] = useState({});
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

  const fetchMetrics = async () => {
    try {
      const response = await axios.get('https://localhost:8000/api/performance', {
        headers: { Authorization: `Bearer ${token}` },
        params: { limit: 10 }
      });
      setMetrics(response.data);
      setMessage('Metrics fetched successfully');
    } catch (error) {
      setMessage(`Fetch metrics failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const fetchHealth = async () => {
    try {
      const response = await axios.get('https://localhost:8000/health');
      setHealth(response.data);
      setMessage('Health status fetched successfully');
    } catch (error) {
      setMessage(`Fetch health failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  useEffect(() => {
    if (token) {
      fetchMetrics();
      fetchHealth();
    }
  }, [token]);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Vial MCP Dashboard</h1>
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
        <button onClick={fetchMetrics} className="bg-purple-500 text-white p-2 rounded mr-2">
          Refresh Metrics
        </button>
        <button onClick={fetchHealth} className="bg-green-500 text-white p-2 rounded">
          Refresh Health
        </button>
      </div>
      <div className="mb-4">
        <h2 className="text-xl font-semibold">System Health</h2>
        <ul className="list-disc pl-5">
          {Object.entries(health.services || {}).map(([service, status]) => (
            <li key={service} className={status === "healthy" ? "text-green-500" : "text-red-500"}>
              {service}: {status}
            </li>
          ))}
          <li className={health.overall === "healthy" ? "text-green-500" : "text-red-500"}>
            Overall: {health.overall || "N/A"}
          </li>
        </ul>
      </div>
      <div>
        <h2 className="text-xl font-semibold">Performance Metrics</h2>
        <table className="table-auto w-full border">
          <thead>
            <tr>
              <th className="border px-4 py-2">Endpoint</th>
              <th className="border px-4 py-2">Wallet ID</th>
              <th className="border px-4 py-2">Response Time (s)</th>
              <th className="border px-4 py-2">Status Code</th>
              <th className="border px-4 py-2">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric, index) => (
              <tr key={index}>
                <td className="border px-4 py-2">{metric.endpoint}</td>
                <td className="border px-4 py-2">{metric.wallet_id}</td>
                <td className="border px-4 py-2">{metric.response_time.toFixed(3)}</td>
                <td className="border px-4 py-2">{metric.status_code}</td>
                <td className="border px-4 py-2">{metric.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {message && <p className="text-red-500">{message}</p>}
    </div>
  );
};

ReactDOM.render(<DashboardPage />, document.getElementById('root'));
