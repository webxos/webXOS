// main/app/tools/page.tsx
'use client';

import { useState, useEffect, useCallback } from 'react';
import { trace } from '@opentelemetry/api';
import styles from './tools.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';

interface Tool {
  id: string;
  name: string;
  status: 'active' | 'inactive';
  description: string;
}

const Tools = () => {
  const [tools, setTools] = useState<Tool[]>([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const tracer = trace.getTracer('vial_mcp_tools');

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

  const fetchTools = useCallback(async () => {
    const span = tracer.startSpan('fetch_tools');
    try {
      const response = await fetchWithRetry(`${API_BASE}/tools`, { method: 'GET' });
      const data = await response.json();
      setTools(data.tools || []);
      setIsAuthenticated(true);
      span.setStatus({ code: 1 });
    } catch (error: any) {
      setErrorMessage(`Failed to load tools: ${error.message}`);
      span.setStatus({ code: 2 });
      span.recordException(error);
    } finally {
      span.end();
    }
  }, [fetchWithRetry]);

  const toggleToolStatus = useCallback(async (toolId: string, currentStatus: string) => {
    const span = tracer.startSpan('toggle_tool_status', { attributes: { toolId } });
    try {
      const newStatus = currentStatus === 'active' ? 'inactive' : 'active';
      const response = await fetchWithRetry(`${API_BASE}/tools/${toolId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      });
      const data = await response.json();
      setTools(prev => prev.map(tool => tool.id === toolId ? { ...tool, status: newStatus } : tool));
      setErrorMessage('Tool status updated successfully');
      span.setStatus({ code: 1 });
    } catch (error: any) {
      setErrorMessage(`Failed to update tool status: ${error.message}`);
      span.setStatus({ code: 2 });
      span.recordException(error);
    } finally {
      span.end();
    }
  }, [fetchWithRetry]);

  useEffect(() => {
    if (localStorage.getItem('apiKey') && localStorage.getItem('userId')) {
      fetchTools();
    }
  }, [fetchTools]);

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Tools</h1>
      {errorMessage && (
        <div className={`${styles.error} ${errorMessage.includes('success') ? styles.success : ''}`}>
          {errorMessage}
        </div>
      )}
      <div className={styles.toolGrid}>
        {tools.map(tool => (
          <div key={tool.id} className={styles.toolCard}>
            <span className={styles.toolName}>{tool.name}</span>
            <span className={`${styles.toolStatus} ${tool.status === 'active' ? styles.active : styles.inactive}`}>
              Status: {tool.status}
            </span>
            <span className={styles.toolDescription}>{tool.description}</span>
            <button
              onClick={() => toggleToolStatus(tool.id, tool.status)}
              className={styles.toggleButton}
              disabled={!isAuthenticated}
            >
              Toggle Status
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Tools;
