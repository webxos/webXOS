export class UIFeedback {
    constructor() {
        this.consoleDiv = document.getElementById('console');
    }

    showFeedback(message, type = "info") {
        if (this.consoleDiv) {
            const color = type === "error" ? "#ff0000" : "#00ff00";
            this.consoleDiv.innerHTML += `<p style="color: ${color}; text-shadow: 0 0 5px ${color}">[${type.toUpperCase()}]: ${message}</p>`;
            this.consoleDiv.scrollTop = this.consoleDiv.scrollHeight;
        }
    }

    clearFeedback() {
        if (this.consoleDiv) {
            this.consoleDiv.innerHTML = '';
            this.consoleDiv.scrollTop = this.consoleDiv.scrollHeight;
        }
    }
}

export const uiFeedback = new UIFeedback();

# xAI Artifact Tags: #vial2 #mcp #client #ui #feedback #neon_mcp
