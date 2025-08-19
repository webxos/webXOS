export class Wallet {
    constructor() {
        this.address = null;
        this.balance = 0.0;
        this.hash = null;
    }

    async sync() {
        try {
            const response = await fetch('/vial/wallet/balance');
            const data = await response.json();
            if (data.result) {
                this.address = data.result.data.address;
                this.balance = data.result.data.balance;
                return true;
            }
            return false;
        } catch (e) {
            console.error(`Wallet sync failed: ${e.message}`);
            return false;
        }
    }
}

export const wallet = new Wallet();

# xAI Artifact Tags: #vial2 #mcp #client #wallet #javascript #neon_mcp
