describe('Vial2 Frontend Tests', () => {
    beforeEach(() => {
        document.body.innerHTML = '<div id="console"></div><textarea id="prompt-input"></textarea>';
    });

    test('Error handling displays correctly', () => {
        const { errorHandler } = require('../mcp/client/error_handler.js');
        errorHandler.logError('Test error', 'stack trace');
        const consoleDiv = document.getElementById('console');
        expect(consoleDiv.innerHTML).toContain('Test error');
    });

    test('Command input works', () => {
        const promptInput = document.getElementById('prompt-input');
        promptInput.value = '/help';
        promptInput.dispatchEvent(new Event('keydown', { key: 'Enter' }));
        const consoleDiv = document.getElementById('console');
        expect(consoleDiv.innerHTML).toContain('Available commands');
    });
});

# xAI Artifact Tags: #vial2 #tests #mcp #frontend #javascript #neon_mcp
