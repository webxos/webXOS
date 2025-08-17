import { config } from './config.js';
import { UIUpdater } from './ui-updater.js';

export class OfflineStorage {
  constructor(gateway) {
    this.gateway = gateway;
    this.uiUpdater = new UIUpdater(gateway);
    this.dbName = 'VialMCP';
    this.storeName = 'cache';
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);
      request.onupgradeneeded = event => {
        this.db = event.target.result;
        this.db.createObjectStore(this.storeName, { keyPath: 'key' });
      };
      request.onsuccess = event => {
        this.db = event.target.result;
        this.uiUpdater.addTerminalMessage('Offline storage initialized', 'command');
        resolve();
      };
      request.onerror = event => {
        this.uiUpdater.addTerminalMessage(`Offline storage error: ${event.target.error}`, 'error');
        reject(event.target.error);
      };
    });
  }

  async saveData(key, data) {
    try {
      const tx = this.db.transaction([this.storeName], 'readwrite');
      const store = tx.objectStore(this.storeName);
      await store.put({ key, data });
      this.uiUpdater.addTerminalMessage(`Saved ${key} to offline storage`, 'command');
    } catch (error) {
      this.uiUpdater.addTerminalMessage(`Save error: ${error.message}`, 'error');
    }
  }

  async getData(key) {
    try {
      const tx = this.db.transaction([this.storeName], 'readonly');
      const store = tx.objectStore(this.storeName);
      const request = await store.get(key);
      return request?.data || config[key.toUpperCase()] || null;
    } catch (error) {
      this.uiUpdater.addTerminalMessage(`Retrieve error: ${error.message}`, 'error');
      return null;
    }
  }

  async cacheResponse(endpoint, response) {
    await this.saveData(endpoint, response);
  }

  async getCachedResponse(endpoint) {
    return await this.getData(endpoint);
  }
}
