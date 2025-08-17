import { config } from './config.js';
import { UIUpdater } from './ui-updater.js';
import { VialManager } from './vial-manager.js';
import { TransactionManager } from './transaction-manager.js';

export class CommandProcessor {
  constructor(client, gateway) {
    this.client = client;
    this.gateway = gateway;
    this.uiUpdater = new UIUpdater(gateway);
    this.vialManager = new VialManager(client, gateway);
    this.transactionManager = new TransactionManager(client, gateway);
  }

  async processCommand(command) {
    command = command.trim();
    this.uiUpdater.addTerminalMessage(`Executing: ${command}`, 'command');

    switch (command) {
      case '/help':
        this.uiUpdater.addTerminalMessage('Available commands: /auth, /void, /troubleshoot, /quantum_link, /export, /import, /api_access, /profile, /transactions', 'command');
        break;
      case '/auth':
        await this.gateway.authenticate();
        break;
      case '/void':
        await this.gateway.void();
        break;
      case '/troubleshoot':
        await this.gateway.troubleshoot();
        break;
      case '/quantum_link':
        await this.gateway.quantumLink();
        break;
      case '/export':
        await this.gateway.exportVials();
        break;
      case '/import':
        this.gateway.elements.fileInput.click();
        break;
      case '/api_access':
        await this.gateway.showApiCredentials();
        break;
      case '/profile':
        await this.fetchProfile();
        break;
      case '/transactions':
        await this.transactionManager.fetchTransactionHistory().then(transactions => this.transactionManager.displayTransactions(transactions));
        break;
      default:
        this.uiUpdater.addTerminalMessage(`Unknown command: ${command}`, 'error');
    }
  }

  async fetchProfile() {
    try {
      this.uiUpdater.addTerminalMessage('Fetching profile...', 'command');
      const data = await this.client.request('/profile');
      this.uiUpdater.addTerminalMessage(
        `Profile: User ID: ${data.user_id}, Username: ${data.username || 'N/A'}, Balance: ${data.balance}, Reputation: ${data.reputation}`,
        'command'
      );
    } catch (error) {
      this.uiUpdater.addTerminalMessage(`Profile fetch failed: ${error.message}`, 'error');
    }
  }
}
