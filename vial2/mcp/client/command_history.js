export class CommandHistory {
    constructor() {
        this.history = [];
        this.consoleDiv = document.getElementById('console');
        this.historyIndex = -1;
    }

    addCommand(command) {
        this.history.push(command);
        this.historyIndex = this.history.length;
        if (this.consoleDiv) {
            this.consoleDiv.innerHTML += `<p style="color: #00ff00">[Command]: ${command}</p>`;
            this.consoleDiv.scrollTop = this.consoleDiv.scrollHeight;
        }
    }

    getPreviousCommand() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            return this.history[this.historyIndex];
        }
        return "";
    }

    getNextCommand() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            return this.history[this.historyIndex];
        }
        return "";
    }
}

export const commandHistory = new CommandHistory();

# xAI Artifact Tags: #vial2 #mcp #client #command #history #neon_mcp
