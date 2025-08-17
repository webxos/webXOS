import { UIUpdater } from './ui-updater.js';

export class Notifications {
  constructor(gateway) {
    this.uiUpdater = new UIUpdater(gateway);
    this.isPermissionGranted = false;
  }

  async requestPermission() {
    try {
      if ('Notification' in window) {
        const permission = await Notification.requestPermission();
        this.isPermissionGranted = permission === 'granted';
        this.uiUpdater.addTerminalMessage(
          this.isPermissionGranted ? 'Notification permission granted' : 'Notification permission denied',
          this.isPermissionGranted ? 'command' : 'error'
        );
      } else {
        this.uiUpdater.addTerminalMessage('Notifications not supported in this browser', 'error');
      }
    } catch (error) {
      this.uiUpdater.addTerminalMessage(`Notification error: ${error.message}`, 'error');
    }
  }

  sendNotification(title, options = {}) {
    if (this.isPermissionGranted && 'Notification' in window) {
      new Notification(title, {
        body: options.body || 'Vial MCP Notification',
        icon: '/icon-192.png',
        badge: '/icon-192.png',
        ...options
      });
      this.uiUpdater.addTerminalMessage(`Notification sent: ${title}`, 'command');
    } else {
      this.uiUpdater.addTerminalMessage(`Cannot send notification: ${title}`, 'error');
    }
  }

  notifyVialStatus(vialId, status) {
    this.sendNotification(`Vial ${vialId} Status Update`, {
      body: `Vial ${vialId} is now ${status}`,
      data: { vialId, status }
    });
  }

  notifyTransaction(transactionId, amount) {
    this.sendNotification('Transaction Completed', {
      body: `Transaction ${transactionId} for ${amount} $WEBXOS completed`,
      data: { transactionId, amount }
    });
  }
}
