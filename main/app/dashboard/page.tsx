import React, { useState, useEffect } from 'react';
import styles from './dashboard.module.css';

interface Metric {
  cpu_usage: number;
  memory_usage: number;
  active_users: number;
  balance: number;
}

interface Vial {
  name: string;
  balance: number;
}

const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric | null>(null);
  const [vials, setVials] = useState<Vial[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = localStorage.getItem('access_token');
        if (!token) throw new Error('No authentication token found');
        const response = await fetch('/mcp/status', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
          body: JSON.stringify({
            jsonrpc: '2.0',
            method: 'mcp.getSystemMetrics',
            params: { user_id: 'test_user' },
            id: 1,
          }),
        });
        const data = await response.json();
        if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);
        setMetrics(data.result);
        const cached = JSON.parse(await localStorage.getItem('vials') || '{}');
        setVials(Object.entries(cached).map(([name, { balance }]) => ({ name, balance })));
      } catch (err: any) {
        setError(`Failed to fetch data: ${err.message}\n${err.stack}`);
        console.error('Dashboard error:', err);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const handleExport = async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) throw new Error('No authentication token found');
      const response = await fetch('/mcp/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'mcp.exportMd', params: { user_id: 'test_user' }, id: 3 }),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      const blob = new Blob([data.result], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'dashboard_data.md';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(`Export failed: ${err.message}\n${err.stack}`);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Dashboard</h1>
      {error && <div className={styles.error}><pre>{error}</pre></div>}
      {metrics && (
        <div className={styles.metrics}>
          <div>CPU Usage: {metrics.cpu_usage}%</div>
          <div>Memory Usage: {metrics.memory_usage}%</div>
          <div>Active Users: {metrics.active_users}</div>
          <div>WebXOS Balance: {metrics.balance}</div>
        </div>
      )}
      <div className={styles.vials}>
        {vials.map((vial) => (
          <div key={vial.name} className={styles.vialCard}>
            <h2>{vial.name}</h2>
            <p>Balance: {vial.balance} WebXOS</p>
          </div>
        ))}
      </div>
      <button className={styles.button} onClick={handleExport}>Export MD</button>
    </div>
  );
};

export default DashboardPage;
