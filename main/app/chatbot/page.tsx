// main/app/chatbot/page.tsx
'use client';

import { useState } from 'react';
import styles from './chatbot.module.css';
import { login } from '../../server/mcp/functions/auth';
import { createNote } from '../../server/mcp/functions/notes';

export default function Chatbot() {
  const [query, setQuery] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('apiKey'));

  const handleSearch = async () => {
    if (!isAuthenticated) {
      alert('Please authenticate first.');
      return;
    }
    try {
      const note = await createNote('Search Query', query, ['search']);
      alert(`Search query saved as note: ${note.note_id}`);
    } catch (error) {
      console.error('Search failed:', error);
      alert('Failed to process search.');
    }
  };

  const handleClear = () => {
    setQuery('');
  };

  const handleAuthenticate = async () => {
    try {
      const username = prompt('Enter username:');
      const password = prompt('Enter password:');
      if (username && password) {
        await login(username, password);
        setIsAuthenticated(true);
        alert('Authentication successful!');
      }
    } catch (error) {
      console.error('Authentication failed:', error);
      alert('Authentication failed.');
    }
  };

  const handleImport = () => {
    alert('Import functionality not yet implemented.');
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>WebXOS Searchbot with Vial</h1>
      </header>
      <main className={styles.main}>
        <div className={styles.searchContainer}>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Search..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className={styles.buttonGroup}>
            <button className={styles.button} onClick={handleSearch}>
              Search
            </button>
            <button className={styles.button} onClick={handleClear}>
              Clear
            </button>
            <button className={styles.button} onClick={handleAuthenticate}>
              Authenticate
            </button>
            <button className={styles.button} onClick={handleImport}>
              Import
            </button>
          </div>
        </div>
      </main>
      <footer className={styles.footer}>
        <p>Copyright webXOS 2025</p>
      </footer>
    </div>
  );
}
