'use client';

import { useEffect } from 'react';
import Head from 'next/head';

export default function Dashboard() {
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token || !JSON.parse(localStorage.getItem('lastResponse') || '{}').vials) {
      window.location.href = '/vial';
    }
  }, []);

  return (
    <div style={{ background: '#000', color: '#0f0', minHeight: '100vh', padding: '1rem', textAlign: 'center' }}>
      <Head>
        <title>Vial MCP Dashboard</title>
      </Head>
      <h1>DASHBOARD</h1>
      <p>Welcome to your Vial MCP Dashboard. Your vials: {JSON.parse(localStorage.getItem('lastResponse') || '{}').vials?.join(', ') || 'None'}</p>
    </div>
  );
}
