import Fuse from '/static/fuse.min.js';

const WebXOSSearch = {
  logs: [],
  fuse: null,

  async init(indexPath) {
    try {
      console.log(`[WebXOSSearch] Initializing search with index at ${indexPath}`);
      this.logs = [{ message: 'Initial log', timestamp: new Date().toLocaleTimeString() }];
      this.fuse = new Fuse(this.logs, {
        keys: ['message'],
        threshold: 0.4
      });
      console.log('[WebXOSSearch] Search initialized');
      return true;
    } catch (err) {
      console.error(`[WebXOSSearch] Initialization failed: ${err.message}`);
      throw err;
    }
  },

  search(query) {
    if (!this.fuse) return [];
    return this.fuse.search(query);
  },

  updateLog(log) {
    this.logs.push(log);
    this.fuse = new Fuse(this.logs, {
      keys: ['message'],
      threshold: 0.4
    });
  },

  clearLogs() {
    this.logs = [];
    this.fuse = new Fuse(this.logs, {
      keys: ['message'],
      threshold: 0.4
    });
  }
};

export { WebXOSSearch };
