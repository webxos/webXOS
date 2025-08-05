window.WebXOSSearch = {
    logs: [],
    fuse: null,
    async init(indexPath) {
        try {
            const response = await fetch(indexPath);
            if (!response.ok) throw new Error(`Failed to load ${indexPath}`);
            const index = await response.json();
            this.logs = index.logs || [];
            if (window.Fuse) {
                this.fuse = new Fuse(this.logs, {
                    keys: ['message', 'timestamp'],
                    threshold: 0.3
                });
            }
            return true;
        } catch (err) {
            console.error(`Search Init Error: ${err.message}`);
            return false;
        }
    },
    updateLog(log) {
        this.logs.push(log);
        if (this.fuse) {
            this.fuse.setCollection(this.logs);
        }
    },
    search(query) {
        if (!this.fuse) return [];
        return this.fuse.search(query);
    },
    clearLogs() {
        this.logs = [];
        if (this.fuse) {
            this.fuse.setCollection(this.logs);
        }
    }
};
