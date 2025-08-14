// main/app/dashboard/page.tsx
'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAccount } from 'wagmi';
import { trace } from '@opentelemetry/api';
import styles from './dashboard.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';

interface Vial {
  id: string;
  status: 'stopped' | 'running';
  tasks: string[];
  latency: number;
}

interface Wallet {
  address: string | null;
  balance: number;
  webxos: number;
}

const Dashboard = () => {
  const [vials, setVials] = useState<Vial[]>([]);
  const [wallet, setWallet] = useState<Wallet>({ address: null, balance: 0, webxos: 0 });
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const { address } = useAccount();
  const tracer = trace.getTracer('vial_mcp_dashboard');

  const fetchWithRetry = useCallback(async (url: string, options: RequestInit, retries = 3, baseDelay = 1000): Promise<{ ok: boolean; json: () => Promise<any> }> => {
    const span = tracer.startSpan('fetch_with_retry', { attributes: { url, retries } });
    const apiKey = localStorage.getItem('apiKey');
    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, { ...options, headers: { ...options.headers, 'Authorization': `Bearer ${apiKey}` } });
        span.setStatus({ code: 1 });
        span.end();
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        return { ok: response.ok, json: () => response.json() };
      } catch (error: any) {
        if (i === retries - 1) {
          span.recordException(error);
          span.setStatus({ code: 2 });
          span.end();
          setErrorMessage(`Fetch failed: ${error.message}`);
          return { ok: false, json: () => ({ error: 'Offline mode' }) };
        }
        await new Promise(resolve => setTimeout(resolve, baseDelay * Math.pow(2, i)));
      }
    }
    span.end();
    return { ok: false, json: () => ({ error: 'Fetch failed' }) };
  }, [tracer]);

  const fetchDashboardData = useCallback(async () => {
    const span = tracer.startSpan('fetch_dashboard_data');
    try {
      const [vialsResponse, walletResponse] = await Promise.all([
        fetchWithRetry(`${API_BASE}/vials/status`, { method: 'GET' }),
        fetchWithRetry(`${API_BASE}/wallet/verify?user_id=${localStorage.getItem('userId')}&address=${address || ''}`, { method: 'GET' })
      ]);
      const vialsData = await vialsResponse.json();
      const walletData = await walletResponse.json();
      setVials(vialsData.vials || []);
      setWallet(walletData || { address: null, balance: 0, webxos: 0 });
      setIsAuthenticated(true);
      span.setStatus({ code: 1 });
    } catch (error: any) {
      setErrorMessage(`Failed to load dashboard: ${error.message}`);
      span.setStatus({ code: 2 });
      span.recordException(error);
    } finally {
      span.end();
    }
  }, [fetchWithRetry, address]);

  useEffect(() => {
    if (localStorage.getItem('apiKey') && localStorage.getItem('userId')) {
      fetchDashboardData();
      const interval = setInterval(fetchDashboardData, 30000);
      return () => clearInterval(interval);
    }
  }, [fetchDashboardData]);

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Dashboard</h1>
      {errorMessage && (
        <div className={`${styles.error} ${errorMessage.includes('success') ? styles.success : ''}`}>
          {errorMessage}
        </div>
      )}
      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Vial Status</h2>
        <div className={styles.vialGrid}>
          {vials.map(vial => (
            <div key={vial.id} className={styles.vialCard}>
              <span className={styles.vialId}>{vial.id}</span>
              <span className={`${styles.vialStatus} ${vial.status === 'running' ? styles.running : styles.stopped}`}>
                Status: {vial.status}
              </span>
              <span className={styles.vialDetails}>Tasks: {vial.tasks.join(', ') || 'None'}</span>
              <span className={styles.vialDetails}>Latency: {vial.latency}ms</span>
            </div>
          ))}
        </div>
      </div>
      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Wallet</h2>
        <div className={styles.walletInfo}>
          <span>Address: {wallet.address || 'Not connected'}</span>
          <span>Balance: {wallet.balance} ETH</span>
          <span>WEBXOS: {wallet.webxos.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
