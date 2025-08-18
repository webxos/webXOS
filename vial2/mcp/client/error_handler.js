export class ErrorHandler {
    constructor() {
        this.errors = [];
    }

    logError(message, stack) {
        this.errors.push({ message, stack, timestamp: new Date().toISOString() });
        console.error(`Error: ${message}`, stack);
        this.displayError(message);
    }

    displayError(message) {
        const consoleDiv = document.getElementById('console');
        if (consoleDiv) {
            const errorMsg = `<p class="error">[${new Date().toISOString()}] ${message}</p>`;
            consoleDiv.innerHTML += errorMsg;
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        }
    }

    clearErrors() {
        this.errors = [];
        const consoleDiv = document.getElementById('console');
        if (consoleDiv) consoleDiv.innerHTML = '';
    }
}

export const errorHandler = new ErrorHandler();

# xAI Artifact Tags: #vial2 #mcp #client #error #handler #neon_mcp
