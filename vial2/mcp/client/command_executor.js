export class CommandExecutor {
    constructor() {
        this.consoleDiv = document.getElementById('console');
        this.commandHistory = null;
    }

    setHistory(history) {
        this.commandHistory = history;
    }

    async executeCommand(command) {
        try {
            if (this.commandHistory) this.commandHistory.addCommand(command);
            const response = await fetch('/mcp/api/vial/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: command })
            });
            const data = await response.json();
            if (data.result) {
                this.displayResponse(data.result.data);
                return true;
            }
            return false;
        } catch (e) {
            console.error(`Command execution failed: ${e.message}`);
            this.displayResponse(`[Error]: ${e.message}`);
            return false;
        }
    }

    displayResponse(response) {
        if (this.consoleDiv) {
            this.consoleDiv.innerHTML += `<p style="color: #00ff00">[Response]: ${response}</p>`;
            this.consoleDiv.scrollTop = this.consoleDiv.scrollHeight;
        }
    }
}

export const commandExecutor = new CommandExecutor();

# xAI Artifact Tags: #vial2 #mcp #client #command #executor #neon_mcp
