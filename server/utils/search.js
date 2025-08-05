console.log('[WebXOSSearch] Loading search.js at /server/utils/search.js');
import Fuse from '/static/fuse.min.js';

const WebXOSSearch = {
  logs: [],
  fuse: null,

  async init(indexPath) {
    console.log(`[WebXOSSearch] Initializing with index at ${indexPath}`);
    this.logs = [{ message: 'Initial log', timestamp: new Date().toLocaleTimeString() }];
    this.fuse = new Fuse(this.logs, {
      keys: ['message'],
      threshold: 0.4
    });
    console.log('[WebXOSSearch] Initialization complete');
    return true;
  },

  search(query) {
    console.log(`[WebXOSSearch] Searching for: ${query}`);
    if (!this.fuse) return [];
    return this.fuse.search(query);
  },

  updateLog(log) {
    console.log('[WebXOSSearch] Updating log:', log);
    this.logs.push(log);
    this.fuse = new Fuse(this.logs, {
      keys: ['message'],
      threshold: 0.4
    });
  },

  clearLogs() {
    console.log('[WebXOSSearch] Clearing logs');
    this.logs = [];
    this.fuse = new Fuse(this.logs, {
      keys: ['message'],
      threshold: 0.4
    });
  }
};

window.WebXOSSearch = WebXOSSearch;
export { WebXOSSearch };
