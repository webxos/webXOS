// main/app/resources/page.tsx
'use client';

import { useState, useEffect } from 'react';
import styles from './resources.module.css';
import { listResources, getResourceById } from '../../server/mcp/functions/resources';
import { login } from '../../server/mcp/functions/auth';

export default function Resources() {
  const [resources, setResources] = useState([]);
  const [selectedResource, setSelectedResource] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('apiKey'));

  useEffect(() => {
    if (isAuthenticated) {
      fetchResources();
    }
  }, [isAuthenticated]);

  const fetchResources = async () => {
    try {
      const data = await listResources('docs');
      setResources(data);
    } catch (error) {
      console.error('Failed to fetch resources:', error);
      alert('Failed to load resources.');
    }
  };

  const handleResourceClick = async (resourceId) => {
    try {
      const resource = await getResourceById(resourceId);
      setSelectedResource(resource);
    } catch (error) {
      console.error('Failed to fetch resource:', error);
      alert('Failed to load resource details.');
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
        fetchResources();
      }
    } catch (error) {
      console.error('Authentication failed:', error);
      alert('Authentication failed.');
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>WebXOS Resources with Vial</h1>
      </header>
      <main className={styles.main}>
        <div className={styles.resourcesContainer}>
          {isAuthenticated ? (
            <>
              <h2>Resources</h2>
              <ul className={styles.resourceList}>
                {resources.map((resource) => (
                  <li
                    key={resource.resource_id}
                    className={styles.resourceItem}
                    onClick={() => handleResourceClick(resource.resource_id)}
                  >
                    {resource.title}
                  </li>
                ))}
              </ul>
              {selectedResource && (
                <div className={styles.resourceDetails}>
                  <h3>{selectedResource.title}</h3>
                  <p>{selectedResource.content}</p>
                </div>
              )}
              <button className={styles.button} onClick={fetchResources}>
                Refresh Resources
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
