'use client';

import { useState, useEffect } from 'react';
import Head from 'next/head';

export default function Vial() {
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string>('');

  const SERVER_URL = 'https://webxos.netlify.app/vial2/api';

  const fetchWithFallback = async (url: string) => {
    try {
      const response = await fetch(url, { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const contentType = response.headers.get('content-type');
      if (!contentType?.includes('application/json')) {
        const text = await response.text();
        throw new Error(`Expected JSON, received: ${text.substring(0, 100)}`);
      }
      return await response.json();
    } catch (err) {
      throw err instanceof Error ? err : new Error(String(err));
    }
  };

  const troubleshoot = async () => {
    setLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] Running troubleshoot...`]);
    try {
      const result = await fetchWithFallback(`${SERVER_URL}/troubleshoot`);
      setLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] Troubleshoot Report: ${JSON.stringify(result)}`]);
    } catch (err) {
      setLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] Troubleshoot Error: ${err.message}`]);
      setError(`Troubleshoot Failed: ${err.message}`);
    }
  };

  const oauthAuthenticate = async () => {
    setLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] Running authentication...`]);
    try {
      const result = await fetchWithFallback(`${SERVER_URL}/auth/oauth`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: 'mock', code: 'test_code' })
      });
      setLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] Authentication successful.`]);
      // Unlock dashboard logic here
    } catch (err) {
      setLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] Authentication Error: ${err.message}`]);
      setError(`Authentication Failed: ${err.message}`);
    }
  };

  return (
    <div style={{ background: '#000', color: '#0f0', minHeight: '100vh', padding: '1rem' }}>
      <Head>
        <title>Vial MCP Gateway</title>
      </Head>
      <h1 style={{ textAlign: 'center', margin: '1rem 0' }}>VIAL MCP GATEWAY</h1>
      <div style={{ maxWidth: '900px', margin: '0 auto', overflowY: 'auto', maxHeight: '60vh' }}>
        {log.map((msg, index) => <p key={index} style={{ margin: '0.5rem 0' }}>{msg}</p>)}
      </div>
      <div style={{ color: '#ff0000', textAlign: 'center', margin: '1rem 0' }}>{error}</div>
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', margin: '1rem 0' }}>
        <button onClick={troubleshoot} style={{ background: '#0f0', color: '#000', padding: '0.5rem 1rem' }}>Troubleshoot</button>
        <button onClick={oauthAuthenticate} style={{ background: '#0f0', color: '#000', padding: '0.5rem 1rem' }}>OAuth</button>
        <button disabled style={{ background: '#666', color: '#000', padding: '0.5rem 1rem' }}>Dashboard</button>
      </div>
    </div>
  );
}
