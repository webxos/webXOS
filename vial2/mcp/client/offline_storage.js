export class OfflineStorage {
    constructor() {
        this.storage = window.localStorage;
    }

    saveState(key, value) {
        try {
            this.storage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.error(`Offline storage save failed: ${e.message}`);
        }
    }

    loadState(key) {
        try {
            const data = this.storage.getItem(key);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error(`Offline storage load failed: ${e.message}`);
            return null;
        }
    }

    clearState() {
        try {
            this.storage.clear();
        } catch (e) {
            console.error(`Offline storage clear failed: ${e.message}`);
        }
    }
}

export const offlineStore = new OfflineStorage();

# xAI Artifact Tags: #vial2 #mcp #client #offline #storage #neon_mcp
