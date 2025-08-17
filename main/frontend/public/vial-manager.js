import { config } from './config.js';
import { formatBalance, handleError } from './utils.js';

export class VialManager {
  constructor(client, gateway) {
    this.client = client;
    this.gateway = gateway;
  }

  async startVial(vialId) {
    try {
      this.client.log(`Starting vial ${vialId}...`);
      const data = await this.client.request('/quantum-link', 'POST', {
        vial_id: vialId,
        work: `work_${vialId}_${Date.now()}`
      });
      const vial = this.gateway.vials.find(v => v.id === vialId);
      if (vial) {
        vial.status = 'Training';
        vial.balance += data.reward || 10;
        vial.wallet.balance = vial.balance;
        this.gateway.updateUI();
        setTimeout(() => {
          vial.status = 'Running';
          this.gateway.updateUI();
          this.client.log(`Vial ${vialId} started successfully`);
        }, 1000);
      }
    } catch (error) {
      handleError(error, this.client.log.bind(this.client));
    }
  }

  async stopVial(vialId) {
    try {
      this.client.log(`Stopping vial ${vialId}...`);
      const vial = this.gateway.vials.find(v => v.id === vialId);
      if (vial) {
        vial.status = 'Stopped';
        this.gateway.updateUI();
        this.client.log(`Vial ${vialId} stopped successfully`);
      }
    } catch (error) {
      handleError(error, this.client.log.bind(this.client));
    }
  }

  async getVialStatus(vialId) {
    try {
      const data = await this.client.request('/health');
      const vial = data.vials.find(v => v.id === vialId) || config.VIALS.find(v => v === vialId);
      return {
        id: vialId,
        status: vial?.status || 'Stopped',
        balance: formatBalance(vial?.balance || 0)
      };
    } catch (error) {
      handleError(error, this.client.log.bind(this.client));
      return config.VIALS.find(v => v === vialId);
    }
  }
}
