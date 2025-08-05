const WebXOSSearch = {
    fuse: null,
    logs: [],

    async init(indexPath) {
        try {
            const response = await fetch(indexPath);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${indexPath}: HTTP ${response.status}`);
            }
            const indexData = await response.json();
            if (window.Fuse) {
                this.fuse = new Fuse(indexData.concat(this.logs), {
                    keys: ['message', 'timestamp', 'agent'],
                    threshold: 0.3,
                    includeScore: true
                });
            } else {
                throw new Error('Fuse.js not available. Ensure /static/fuse.min.js is loaded.');
            }
        } catch (err) {
            throw new Error(`Search Initialization Error: ${err.message}`);
        }
    },

    updateLog(log) {
        if (this.fuse) {
            this.logs.push(log);
            this.fuse.setCollection(this.logs);
        }
    },

    clearLogs() {
        this.logs = [];
        if (this.fuse) {
            this.fuse.setCollection(this.logs);
        }
    },

    search(query) {
        if (!this.fuse) return [];
        return this.fuse.search(query).slice(0, 10); // Limit to 10 results
    }
};

// Expose WebXOSSearch globally
window.WebXOSSearch = WebXOSSearch;
