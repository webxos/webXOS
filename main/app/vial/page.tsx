import React, { useState, useEffect } from 'react';
import { createSession } from '../../server/mcp/functions/auth';
import styles from './vial.module.css';

interface Resource {
  resource_id: string;
  uri: string;
  metadata: { name: string };
}

const VialPage: React.FC = () => {
  const [resources, setResources] = useState<Resource[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [userId] = useState<string>('test_user'); // Replace with actual user auth

  useEffect(() => {
    const fetchResources = async () => {
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
            method: 'mcp.listResources',
            params: { user_id: userId, agent_id: 'default_agent' },
            id: 1,
          }),
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error.message);
        setResources(data.result);
      } catch (err: any) {
        setError(`Failed to fetch resources: ${err.message}\n${err.stack}`);
        console.error('Vial error:', err);
      }
    };
    fetchResources();
  }, [userId]);

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Controller</h1>
      {error && (
        <div className={styles.error}>
          <pre>{error}</pre>
        </div>
      )}
      <div className={styles.resources}>
        {resources.map((resource) => (
          <div key={resource.resource_id} className={styles.resourceCard}>
            <h2>{resource.metadata.name}</h2>
            <p><a href={resource.uri} target="_blank" rel="noopener noreferrer">{resource.uri}</a></p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default VialPage;
