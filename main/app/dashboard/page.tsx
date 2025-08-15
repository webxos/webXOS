import React, { useState, useEffect } from 'react';
import styles from './dashboard.module.css';

interface Metric {
  name: string;
  value: string | number;
}

interface MCPRequest {
  message: string;
  description: string;
}

const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [userId] = useState<string>('test_user'); // Replace with actual user auth
  const mcpRequests: MCPRequest[] = [
    { message: 'InitializeRequest', description: 'This request is sent from the client to the server when it first connects, asking it to begin initialization' },
    { message: 'ListToolsRequest', description: 'Sent from the client to request a list of tools the server has' },
    { message: 'CallToolRequest', description: 'Used by the client to invoke a tool provided by the server' },
    { message: 'ListResourcesRequest', description: 'Sent from the client to request a list of resources the server has' },
    { message: 'ReadResourceRequest', description: 'Sent from the client to the server, to read a specific resource URI' },
    { message: 'ListPromptsRequest', description: 'Sent from the client to request a list of prompts and prompt templates the server has' },
    { message: 'GetPromptRequest', description: 'Used by the client to get a prompt provided by the server' },
    { message: 'PingRequest', description: 'A ping, issued by either the server or the client, to check that the other party is still alive' },
    { message: 'CreateMessageRequest', description: 'A request from the server to sample an LLM via the client. The client has full discretion over which model to select. The client should also inform the user before beginning sampling, to allow them to inspect the request (human in the loop) and decide whether to approve it' },
    { message: 'SetLevelRequest', description: 'A request from the client to the server, to enable or adjust logging' },
  ];

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/mcp', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
          body: JSON.stringify({
            jsonrpc: '2.0',
            method: 'mcp.getSystemMetrics',
            params: { user_id: userId },
            id: 1,
          }),
        });
        const data = await response.json();
        if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);
        setMetrics([
          { name: 'CPU Usage', value: data.result.cpu_usage },
          { name: 'Memory Usage', value: data.result.memory_usage },
          { name: 'Active Users', value: data.result.active_users },
        ]);
      } catch (err: any) {
        setError(`Failed to fetch metrics: ${err.message}\n${err.stack}`);
        console.error('Dashboard error:', err);
      }
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, [userId]);

  const handleAction = async (action: string) => {
    try {
      let response;
      switch (action) {
        case 'createSession':
          response = await fetch('/mcp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
            body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.createSession', params: { user_id: userId }, id: 2 }),
          });
          break;
        case 'createNote':
          response = await fetch('/mcp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
            body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.createNote', params: { user_id: userId, title: 'Test Note', content: 'Test content' }, id: 3 }),
          });
          break;
        case 'syncResources':
          response = await fetch('/mcp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
            body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.syncResources', params: { user_id: userId, repo_name: 'test/repo' }, id: 4 }),
          });
          break;
        default:
          throw new Error(`Unknown action: ${action}`);
      }
      const data = await response.json();
      if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);
      setMessages((prev) => [...prev, { id: Date.now().toString(), text: `Action ${action} succeeded: ${JSON.stringify(data.result)}`, sender: 'system' }]);
    } catch (err: any) {
      setError(`Action ${action} failed: ${err.message}\n${err.stack}`);
      console.error('Action error:', err);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Dashboard</h1>
      {error && (
        <div className={styles.error}>
          <pre>{error}</pre>
        </div>
      )}
      <div className={styles.metrics}>
        {metrics.map((metric) => (
          <div key={metric.name} className={styles.metricCard}>
            <h2>{metric.name}</h2>
            <p>{metric.value}</p>
          </div>
        ))}
      </div>
      <div className={styles.inspector}>
        <h2>MCP Inspector</h2>
        <table className={styles.inspectorTable}>
          <thead>
            <tr>
              <th>Message</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {mcpRequests.map((req, index) => (
              <tr key={index}>
                <td>{req.message}</td>
                <td>{req.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={styles.buttonGroup}>
        <button onClick={() => handleAction('createSession')} className={styles.button}>Create Session</button>
        <button onClick={() => handleAction('createNote')} className={styles.button}>Create Note</button>
        <button onClick={() => handleAction('syncResources')} className={styles.button}>Sync Resources</button>
      </div>
    </div>
  );
};

export default DashboardPage;
