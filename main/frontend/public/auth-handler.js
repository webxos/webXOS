import { config } from './config.js';
import { UIUpdater } from './ui-updater.js';
import { ErrorHandler } from './error-handler.js';

export class AuthHandler {
  constructor(client, gateway) {
    this.client = client;
    this.gateway = gateway;
    this.uiUpdater = new UIUpdater(gateway);
    this.errorHandler = new ErrorHandler(client, gateway);
  }

  async login(apiKey) {
    try {
      this.uiUpdater.addTerminalMessage('Authenticating...', 'command');
      const data = await this.client.request('/auth', 'POST', { api_key: apiKey });
      localStorage.setItem('access_token', data.access_token);
      this.gateway.isAuthenticated = true;
      this.gateway.updateButtonStates();
      this.uiUpdater.addTerminalMessage('Authentication successful', 'command');
      return data;
    } catch (error) {
      this.errorHandler.handle(error, '/auth');
      return null;
    }
  }

  async register(username) {
    try {
      this.uiUpdater.addTerminalMessage(`Registering user: ${username}...`, 'command');
      const data = await this.client.request('/register', 'POST', { username });
      localStorage.setItem('access_token', data.token);
      this.gateway.isAuthenticated = true;
      this.gateway.updateButtonStates();
      this.uiUpdater.addTerminalMessage(`Registration successful, User ID: ${data.user_id}`, 'command');
      this.uiUpdater.updateCredentials({ key: data.api_key, secret: data.api_secret });
      return data;
    } catch (error) {
      this.errorHandler.handle(error, '/register');
      return null;
    }
  }

  logout() {
    localStorage.removeItem('access_token');
    this.gateway.isAuthenticated = false;
    this.gateway.updateButtonStates();
    this.uiUpdater.updateWallet(config);
    this.uiUpdater.updateVials(config.VIALS.map(id => ({ id, status: 'Stopped', balance: 0 })));
    this.uiUpdater.addTerminalMessage('Logged out successfully', 'command');
  }
}
