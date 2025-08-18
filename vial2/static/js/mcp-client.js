let isAuthenticated = false;
let accessToken = null;

async function sendRequest(endpoint, method, data, auth = true) {
    const headers = { 'Content-Type': 'application/json' };
    if (auth && accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
    }
    try {
        const response = await fetch(endpoint, {
            method,
            headers,
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.error) {
            console.error(`Request failed: ${result.error} [mcp-client.js:10] [ID:request_error]`);
            return { error: result.error };
        }
        return result;
    } catch (e) {
        console.error(`Request failed: ${e.message} [mcp-client.js:15] [ID:request_error]`);
        return { error: e.message };
    }
}

function setAuthenticated(token) {
    isAuthenticated = true;
    accessToken = token;
    ['quantumLinkButton', 'exportButton', 'importButton', 'apiAccessButton', 'prompt-input'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = false;
    });
    document.getElementById('authButton').classList.add('active-monitor');
}

function resetAuthenticated() {
    isAuthenticated = false;
    accessToken = null;
    ['quantumLinkButton', 'exportButton', 'importButton', 'apiAccessButton', 'prompt-input'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = true;
    });
    document.getElementById('authButton').classList.remove('active-monitor');
}

export { sendRequest, setAuthenticated, resetAuthenticated };
