import React, { useState, useEffect } from 'react';
import { createSession } from '../../server/mcp/functions/auth';
import styles from './dashboard.module.css';

interface Metric {
  name: string;
  value: string | number;
}

const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [userId] = useState<string>('test_user'); // Replace with actual user auth

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        await createSession(userId);
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
        if (data.error) throw new Error(data.error.message);
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
  }, [userId]);

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
    </div>
  );
};

export default DashboardPage;
