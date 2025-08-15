// main/app/settings/page.tsx
'use client';

import { useState, useEffect } from 'react';
import styles from './settings.module.css';
import { login, updateUserSettings } from '../../server/mcp/functions/auth';

export default function Settings() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('apiKey'));
  const [settings, setSettings] = useState({ theme: 'dark', notifications: true });

  useEffect(() => {
    if (isAuthenticated) {
      fetchSettings();
    }
  }, [isAuthenticated]);

  const fetchSettings = async () => {
    try {
      // Placeholder: Fetch user settings from API
      setSettings({ theme: 'dark', notifications: true });
    } catch (error) {
      console.error('Failed to fetch settings:', error);
      alert('Failed to load settings.');
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
        fetchSettings();
      }
    } catch (error) {
      console.error('Authentication failed:', error);
      alert('Authentication failed.');
    }
  };

  const handleSaveSettings = async () => {
    try {
      await updateUserSettings(settings);
      alert('Settings saved successfully!');
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('Failed to save settings.');
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>WebXOS Settings with Vial</h1>
      </header>
      <main className={styles.main}>
        <div className={styles.settingsContainer}>
          {isAuthenticated ? (
            <>
              <h2>User Settings</h2>
              <div className={styles.settingsForm}>
                <label>
                  Theme:
                  <select
                    value={settings.theme}
                    onChange={(e) => setSettings({ ...settings, theme: e.target.value })}
                    className={styles.input}
                  >
                    <option value="dark">Dark</option>
                    <option value="light">Light</option>
                  </select>
                </label>
                <label>
                  Notifications:
                  <input
                    type="checkbox"
                    checked={settings.notifications}
                    onChange={(e) => setSettings({ ...settings, notifications: e.target.checked })}
                    className={styles.checkbox}
                  />
                </label>
                <button className={styles.button} onClick={handleSaveSettings}>
                  Save Settings
                </button>
              </div>
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
