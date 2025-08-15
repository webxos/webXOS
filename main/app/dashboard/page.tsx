// main/app/dashboard/page.tsx
'use client';

import { useState, useEffect } from 'react';
import styles from './dashboard.module.css';
import { getSystemMetrics } from '../../server/mcp/functions/resources';
import { login } from '../../server/mcp/functions/auth';

export default function Dashboard() {
  const [metrics, setMetrics] = useState({ cpu_usage: 0, memory_usage: 0 });
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('apiKey'));

  useEffect(() => {
    if (isAuthenticated) {
      fetchMetrics();
    }
  }, [isAuthenticated]);

  const fetchMetrics = async () => {
    try {
      const data = await getSystemMetrics();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      alert('Failed to load dashboard metrics.');
    }
  };

  const handleAuthenticate = async () => {
    try {
      const username = prompt('Enter username:');
      const password = prompt('Enter password:');
      if (username && password) {
        await login(username, password);
        setIsAuthenticated(true);
        alert('Authentication successful!');
        fetchMetrics();
      }
    } catch (error) {
      console.error('Authentication failed:', error);
      alert('Authentication failed.');
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>WebXOS Dashboard with Vial</h1>
      </header>
      <main className={styles.main}>
        <div className={styles.dashboardContainer}>
          {isAuthenticated ? (
            <>
              <h2>System Metrics</h2>
              <div className={styles.metrics}>
                <p>CPU Usage: {metrics.cpu_usage}%</p>
                <p>Memory Usage: {metrics.memory_usage}%</p>
              </div>
              <button className={styles.button} onClick={fetchMetrics}>
                Refresh Metrics
              </button>
            </>
          ) : (
            <button className={styles.button} onClick={handleAuthenticate}>
              Authenticate
            </button>
          )}
        </div>
      </main>
      <footer className={styles.footer}>
        <p>Copyright webXOS 2025</p>
      </footer>
    </div>
  );
}
