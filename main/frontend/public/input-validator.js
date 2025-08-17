import { UIUpdater } from './ui-updater.js';

export class InputValidator {
  constructor(gateway) {
    this.uiUpdater = new UIUpdater(gateway);
  }

  validateCommand(command) {
    const validCommands = ['/auth', '/void', '/troubleshoot', '/quantum_link', '/export', '/import', '/api_access', '/profile', '/transactions'];
    if (!validCommands.includes(command.trim())) {
      this.uiUpdater.addTerminalMessage(`Invalid command: ${command}`, 'error');
      return false;
    }
    return true;
  }

  validateAmount(amount) {
    const parsed = parseFloat(amount);
    if (isNaN(parsed) || parsed <= 0) {
      this.uiUpdater.addTerminalMessage(`Invalid amount: ${amount}`, 'error');
      return false;
    }
    return parsed;
  }

  validateAddress(address) {
    if (!address || typeof address !== 'string' || address.length < 10) {
      this.uiUpdater.addTerminalMessage(`Invalid address: ${address}`, 'error');
      return false;
    }
    return true;
  }

  validateVialId(vialId) {
    const validVials = ['vial1', 'vial2', 'vial3', 'vial4'];
    if (!validVials.includes(vialId)) {
      this.uiUpdater.addTerminalMessage(`Invalid vial ID: ${vialId}`, 'error');
      return false;
    }
    return true;
  }
}
