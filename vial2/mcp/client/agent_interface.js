export class AgentInterface {
    constructor() {
        this.consoleDiv = document.getElementById('console');
        this.agentCredentials = {};
    }

    async promptCredentials(agentType) {
        const credential = prompt(`Enter API key for ${agentType}:`);
        if (credential) {
            this.agentCredentials[agentType] = credential;
            this.consoleDiv.innerHTML += `<p>[System]: Credential set for ${agentType}</p>`;
            this.consoleDiv.scrollTop = this.consoleDiv.scrollHeight;
            return true;
        }
        this.consoleDiv.innerHTML += `<p>[System]: Credential entry cancelled for ${agentType}</p>`;
        return false;
    }

    async sendMessage(agentType, message) {
        try {
            if (!this.agentCredentials[agentType] && !await this.promptCredentials(agentType)) {
                return false;
            }
            const response = await fetch('/mcp/api/vial/agent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ agent_type: agentType, message: { content: message }, credentials: { api_key: this.agentCredentials[agentType] } })
            });
            const data = await response.json();
            if (data.result) {
                this.displayResponse(data.result.data);
                return true;
            }
            return false;
        } catch (e) {
            console.error(`Agent interface failed: ${e.message}`);
            this.consoleDiv.innerHTML += `<p>[Error]: ${e.message}</p>`;
            return false;
        }
    }

    displayResponse(response) {
        if (this.consoleDiv) {
            this.consoleDiv.innerHTML += `<p>[${response.split(':')[0]}]: ${response.split(':').slice(1).join(':')}</p>`;
            this.consoleDiv.scrollTop = this.consoleDiv.scrollHeight;
        }
    }
}

export const agentInterface = new AgentInterface();

# xAI Artifact Tags: #vial2 #mcp #client #agent #interface #neon_mcp
