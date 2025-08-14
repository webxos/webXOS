// main/app/resources/page.tsx
'use client';

import { useState, useEffect, useCallback } from 'react';
import { trace } from '@opentelemetry/api';
import styles from './resources.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';

interface Resource {
  type: string;
  usage: number;
  total: number;
  unit: string;
}

const Resources = () => {
  const [resources, setResources] = useState<Resource[]>([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const tracer = trace.getTracer('vial_mcp_resources');

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

  const fetchResources = useCallback(async () => {
    const span = tracer.startSpan('fetch_resources');
    try {
      const response = await fetchWithRetry(`${API_BASE}/resources`, { method: 'GET' });
      const data = await response.json();
      setResources(data.resources || []);
      setIsAuthenticated(true);
      span.setStatus({ code: 1 });
    } catch (error: any) {
      setErrorMessage(`Failed to load resources: ${error.message}`);
      span.setStatus({ code: 2 });
      span.recordException(error);
    } finally {
      span.end();
    }
  }, [fetchWithRetry]);

  useEffect(() => {
    if (localStorage.getItem('apiKey') && localStorage.getItem('userId')) {
      fetchResources();
      const interval = setInterval(fetchResources, 30000);
      return () => clearInterval(interval);
    }
  }, [fetchResources]);

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Resources</h1>
      {errorMessage && (
        <div className={`${styles.error} ${errorMessage.includes('success') ? styles.success : ''}`}>
          {errorMessage}
        </div>
      )}
      <div className={styles.resourceGrid}>
        {resources.map((resource, index) => (
          <div key={index} className={styles.resourceCard}>
            <span className={styles.resourceType}>{resource.type}</span>
            <div className={styles.progressBar}>
              <div
                className={styles.progressFill}
                style={{ width: `${(resource.usage / resource.total) * 100}%` }}
              ></div>
            </div>
            <span className={styles.resourceDetails}>
              {resource.usage.toFixed(2)} / {resource.total.toFixed(2)} {resource.unit}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Resources;
