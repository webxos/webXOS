export class AgentManager {
    constructor() {
        this.agents = {};
    }

    async updateStatus() {
        try {
            const response = await fetch('/vial/agent/status');
            const data = await response.json();
            if (data.result) {
                this.agents = data.result.data;
                this.renderStatus();
                return true;
            }
            return false;
        } catch (e) {
            console.error(`Agent status update failed: ${e.message}`);
            return false;
        }
    }

    renderStatus() {
        const consoleDiv = document.getElementById('console');
        if (consoleDiv) {
            consoleDiv.innerHTML += `<p>Agent Status: ${JSON.stringify(this.agents)}</p>`;
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        }
    }
}

export const agentManager = new AgentManager();

# xAI Artifact Tags: #vial2 #mcp #client #agent #manager #neon_mcp
