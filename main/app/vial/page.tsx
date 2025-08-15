// main/app/vial/page.tsx
'use client';

import { useState, useEffect, useCallback } from 'react';
import { trace } from '@opentelemetry/api';
import styles from './vial.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';

interface Vial {
  id: string;
  status: 'stopped' | 'running' | 'paused';
  quantum_circuit: string | null;
  last_updated: string;
}

const Vial = () => {
  const [vials, setVials] = useState<Vial[]>([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const tracer = trace.getTracer('vial_mcp_vial');

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

  const fetchVials = useCallback(async () => {
    const span = tracer.startSpan('fetch_vials');
    try {
      const response = await fetchWithRetry(`${API_BASE}/vials`, { method: 'GET' });
      const data = await response.json();
      setVials(data.vials || []);
      setIsAuthenticated(true);
      span.setStatus({ code: 1 });
    } catch (error: any) {
      setErrorMessage(`Failed to load vials: ${error.message}`);
      span.setStatus({ code: 2 });
      span.recordException(error);
    } finally {
      span.end();
    }
  }, [fetchWithRetry]);

  const controlVial = useCallback(async (vialId: string, action: 'start' | 'stop' | 'pause') => {
    const span = tracer.startSpan('control_vial', { attributes: { vialId, action } });
    try {
      const response = await fetchWithRetry(`${API_BASE}/vials/${vialId}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      setVials(prev => prev.map(vial => vial.id === vialId ? { ...vial, status: action === 'start' ? 'running' : action } : vial));
      setErrorMessage(`Vial ${vialId} ${action}ed successfully`);
      span.setStatus({ code: 1 });
    } catch (error: any) {
      setErrorMessage(`Failed to ${action} vial: ${error.message}`);
      span.setStatus({ code: 2 });
      span.recordException(error);
    } finally {
      span.end();
    }
  }, [fetchWithRetry]);

  useEffect(() => {
    if (localStorage.getItem('apiKey') && localStorage.getItem('userId')) {
      fetchVials();
      const interval = setInterval(fetchVials, 30000);
      return () => clearInterval(interval);
    }
  }, [fetchVials]);

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Control</h1>
      {errorMessage && (
        <div className={`${styles.error} ${errorMessage.includes('success') ? styles.success : ''}`}>
          {errorMessage}
        </div>
      )}
      <div className={styles.vialGrid}>
        {vials.map(vial => (
          <div key={vial.id} className={styles.vialCard}>
            <span className={styles.vialId}>{vial.id}</span>
            <span className={`${styles.vialStatus} ${styles[vial.status]}`}>
              Status: {vial.status}
            </span>
            <span className={styles.vialDetails}>Last Updated: {vial.last_updated}</span>
            <span className={styles.vialDetails}>Circuit: {vial.quantum_circuit || 'None'}</span>
            <div className={styles.vialControls}>
              <button
                onClick={() => controlVial(vial.id, 'start')}
                disabled={vial.status === 'running' || !isAuthenticated}
                className={styles.controlButton}
              >
                Start
              </button>
              <button
                onClick={() => controlVial(vial.id, 'pause')}
                disabled={vial.status === 'paused' || !isAuthenticated}
                className={styles.controlButton}
              >
                Pause
              </button>
              <button
                onClick={() => controlVial(vial.id, 'stop')}
                disabled={vial.status === 'stopped' || !isAuthenticated}
                className={styles.controlButton}
              >
                Stop
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Vial;
