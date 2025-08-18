export async function handleTerminalCommand(command, logCallback, isAuthenticated, isOffline) {
    if (!isAuthenticated) {
        logCallback('Not authenticated: Please authenticate first');
        return;
    }
    if (isOffline) {
        logCallback('Offline mode: All commands disabled except /help. Connect online to enable.');
        return;
    }
    const sanitizedCommand = command.replace(/[<>{}\[\];]/g, '');
    const parts = sanitizedCommand.trim().split(' ');
    const cmd = parts[0].toLowerCase();

    if (cmd === '/help') {
        logCallback(`Available commands:
- /mcp status: Check MCP status and connected resources
- /git status: Check git repository status
- /wallet sync <vial>: Sync wallet for specified vial
- /prompt <vial> <text>: Send prompt to vial
- /task <vial> <task>: Assign task to vial
- /config <vial> <key> <value>: Set vial config`);
        return;
    }

    try {
        const response = await fetch('/mcp/api/vial/console/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({ command: sanitizedCommand, vial_id: parts[1] || 'vial1' })
        });
        const data = await response.json();
        if (data.error) {
            logCallback(`Command failed: ${data.error.message}`);
        } else {
            logCallback(`${data.result.data.output}. Connected: ${data.result.data.connected_resources.join(', ') || 'none'}`);
        }
    } catch (err) {
        logCallback(`Command execution failed: ${err.message}`);
    }
}

// xAI Artifact Tags: #vial2 #mcp #client #terminal #commands #javascript #neon_mcp
