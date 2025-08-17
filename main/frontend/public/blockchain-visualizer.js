import { UIUpdater } from './ui-updater.js';
import { config } from './config.js';

export class BlockchainVisualizer {
  constructor(client, gateway) {
    this.client = client;
    this.uiUpdater = new UIUpdater(gateway);
    this.canvas = document.createElement('canvas');
    this.canvas.width = 400;
    this.canvas.height = 200;
    this.canvas.className = 'blockchain-canvas';
    document.querySelector('.container').appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
  }

  async fetchAndVisualize() {
    try {
      this.uiUpdater.addTerminalMessage('Fetching blockchain data...', 'command');
      const data = await this.client.request('/blockchain-sync', 'POST');
      this.visualizeChain(data.syncedBlocks || []);
    } catch (error) {
      this.uiUpdater.addTerminalMessage(`Blockchain fetch error: ${error.message}`, 'error');
      this.visualizeChain([]);
    }
  }

  visualizeChain(blocks) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillStyle = '#00ff00';
    this.ctx.font = '12px monospace';

    const blockWidth = 80;
    const blockHeight = 40;
    const startX = 20;
    const startY = 20;

    blocks.forEach((block, index) => {
      const x = startX + index * (blockWidth + 20);
      const y = startY;

      // Draw block
      this.ctx.strokeStyle = '#00ff00';
      this.ctx.strokeRect(x, y, blockWidth, blockHeight);
      this.ctx.fillText(`Block ${index + 1}`, x + 5, y + 15);
      this.ctx.fillText(`Hash: ${block.hash.slice(0, 8)}...`, x + 5, y + 30);

      // Draw arrow to next block
      if (index < blocks.length - 1) {
        this.ctx.beginPath();
        this.ctx.moveTo(x + blockWidth, y + blockHeight / 2);
        this.ctx.lineTo(x + blockWidth + 20, y + blockHeight / 2);
        this.ctx.stroke();
      }
    });

    this.uiUpdater.addTerminalMessage(`Visualized ${blocks.length} blockchain blocks`, 'command');
  }
}
