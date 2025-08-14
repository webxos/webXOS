// main/app/settings/page.tsx
'use client';

import { useState, useCallback, useEffect } from 'react';
import { trace } from '@opentelemetry/api';
import styles from './settings.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';
const PROVIDERS = ['gemini', 'openai', 'claude', 'oci'];

const Settings = () => {
  const [apiKey, setApiKey] = useState<string | null>(typeof window !== 'undefined' ? localStorage.getItem('apiKey') : null);
  const [userId, setUserId] = useState<string | null>(typeof window !== 'undefined' ? localStorage.getItem('userId') : null);
  const [preferredProvider, setPreferredProvider] = useState(PROVIDERS[0]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const tracer = trace.getTracer('vial_mcp_settings');

  const fetchWithRetry = useCallback(async (url: string, options: RequestInit, retries = 3, baseDelay = 1000): Promise<{ ok: boolean; json: () => Promise<any> }> => {
    const span = tracer.startSpan('fetch_with_retry', { attributes: { url, retries } });
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
  }, [apiKey, tracer]);

  const handleSaveSettings = useCallback(async () => {
    const span = tracer.startSpan('save_settings');
    try {
      if (!isAuthenticated) throw new Error('Authenticate first');
      const response = await fetchWithRetry(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, preferred_provider: preferredProvider })
      });
      const data = await response.json();
      setErrorMessage('Settings saved successfully');
      span.setStatus({ code: 1 });
    } catch (error: any) {
      setErrorMessage(`Failed to save settings: ${error.message}`);
      span.setStatus({ code: 2 });
      span.recordException(error);
    } finally {
      span.end();
    }
  }, [userId, preferredProvider, isAuthenticated, fetchWithRetry]);

  useEffect(() => {
    if (apiKey && userId) {
      setIsAuthenticated(true);
    }
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [apiKey, userId, errorMessage]);

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Settings</h1>
      <div className={styles.form}>
        <label className={styles.label}>User ID</label>
        <input
          type="text"
          value={userId || ''}
          readOnly
          className={styles.input}
          placeholder="Authenticate to view User ID"
        />
        <label className={styles.label}>API Key</label>
        <input
          type="text"
          value={apiKey || ''}
          readOnly
          className={styles.input}
          placeholder="Authenticate to view API Key"
        />
        <label className={styles.label}>Preferred AI Provider</label>
        <select
          value={preferredProvider}
          onChange={(e) => setPreferredProvider(e.target.value)}
          className={styles.select}
          disabled={!isAuthenticated}
        >
          {PROVIDERS.map(p => (
            <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
          ))}
        </select>
        <button onClick={handleSaveSettings} disabled={!isAuthenticated} className={styles.button}>
          Save Settings
        </button>
      </div>
      {errorMessage && (
        <div className={`${styles.error} ${errorMessage.includes('success') ? styles.success : ''}`}>
          {errorMessage}
        </div>
      )}
    </div>
  );
};

export default Settings;
