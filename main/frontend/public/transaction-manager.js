import { config } from './config.js';
import { formatBalance, handleError } from './utils.js';
import { UIUpdater } from './ui-updater.js';

export class TransactionManager {
  constructor(client, gateway) {
    this.client = client;
    this.gateway = gateway;
    this.uiUpdater = new UIUpdater(gateway);
  }

  async initiateTransaction(toAddress, amount) {
    try {
      this.uiUpdater.addTerminalMessage(`Initiating transaction of ${formatBalance(amount)} $WEBXOS to ${toAddress}...`, 'command');
      const data = await this.client.request('/wallet-update', 'POST', {
        to_address: toAddress,
        amount
      });
      this.gateway.wallet.balance -= amount;
      this.uiUpdater.updateWallet(this.gateway.wallet);
      this.uiUpdater.addTerminalMessage(`Transaction ${data.transaction.transaction_id} completed`, 'command');
      return data.transaction;
    } catch (error) {
      handleError(error, this.uiUpdater.addTerminalMessage.bind(this.uiUpdater));
      return null;
    }
  }

  async fetchTransactionHistory() {
    try {
      this.uiUpdater.addTerminalMessage('Fetching transaction history...', 'command');
      const data = await this.client.request('/transactions');
      this.uiUpdater.addTerminalMessage(`Retrieved ${data.transactions.length} transactions`, 'command');
      return data.transactions;
    } catch (error) {
      handleError(error, this.uiUpdater.addTerminalMessage.bind(this.uiUpdater));
      return [];
    }
  }

  displayTransactions(transactions) {
    transactions.forEach(tx => {
      const message = `Transaction ${tx.transaction_id}: ${formatBalance(tx.amount)} $WEBXOS from ${tx.from_address} to ${tx.to_address} at ${tx.created_at}`;
      this.uiUpdater.addTerminalMessage(message, tx.isValid ? 'command' : 'error');
    });
  }
}
