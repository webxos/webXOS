/**
 * Next.js page for Vial MCP chatbot interface.
 */
import { useState, useEffect } from 'https://cdn.jsdelivr.net/npm/react@18.3.1/+esm';
import { useRouter } from 'https://cdn.jsdelivr.net/npm/next@14.2.13/+esm';
import axios from 'https://cdn.jsdelivr.net/npm/axios@1.7.7/+esm';

export default function ChatbotPage() {
    const [apiKey, setApiKey] = useState('');
    const [walletId, setWalletId] = useState('');
    const [accessToken, setAccessToken] = useState('');
    const [prompt, setPrompt] = useState('');
    const [note, setNote] = useState('');
    const [status, setStatus] = useState('Initializing...');
    const [output, setOutput] = useState('');
    const router = useRouter();

    async function logToConsole(message) {
        const timestamp = new Date().toISOString();
        console.log(`[${timestamp}] ${message}`);
        setOutput(prev => prev + `\n[${timestamp}] ${message}`);
    }

    async function authenticate() {
        try {
            if (!apiKey || !walletId) throw new Error("API Key and Wallet ID required");
            const response = await axios.post('https://localhost:8000/api/auth/login', { api_key: apiKey, wallet_id: walletId }, {
                headers: { 'X-API-Key': apiKey }
            });
            setAccessToken(response.data.access_token);
            await logToConsole(`Authenticated with token for wallet ${walletId}`);
            setStatus('Authenticated');
        } catch (e) {
            await logToConsole(`Authentication failed: ${e.message}`);
            setStatus('Authentication Failed');
        }
    }

    async function sendPrompt() {
        try {
            if (!prompt) throw new Error("Prompt required");
            const response = await axios.post('https://localhost:8000/api/quantum/link', 
                { vial_id: 'vial_1', prompt, wallet_id: walletId }, 
                { headers: { 'Authorization': `Bearer ${accessToken}` } }
            );
            await logToConsole(`Prompt sent: ${response.data.message || 'Success'}`);
            setOutput(JSON.stringify(response.data));
        } catch (e) {
            await logToConsole(`Prompt failed: ${e.message}`);
        }
    }

    async function addNote() {
        try {
            if (!note) throw new Error("Note content required");
            const response = await axios.post('https://localhost:8000/api/notes/add', 
                { wallet_id: walletId, content: note }, 
                { headers: { 'Authorization': `Bearer ${accessToken}` } }
            );
            await logToConsole(`Note added: ${response.data.note_id}`);
            setOutput(JSON.stringify(response.data));
        } catch (e) {
            await logToConsole(`Note add failed: ${e.message}`);
        }
    }

    async function getResources() {
        try {
            const response = await axios.post('https://localhost:8000/api/resources/latest', 
                { wallet_id: walletId, limit: 10 }, 
                { headers: { 'Authorization': `Bearer ${accessToken}` } }
            );
            await logToConsole(`Retrieved ${response.data.resources.length} resources`);
            setOutput(JSON.stringify(response.data.resources));
        } catch (e) {
            await logToConsole(`Resource retrieval failed: ${e.message}`);
        }
    }

    useEffect(() => {
        async function checkHealth() {
            try {
                const response = await axios.get('https://localhost:8000/api/health');
                await logToConsole(`Backend online: ${response.data.status}`);
                setStatus('Backend Online');
            } catch (e) {
                await logToConsole(`Health check failed: ${e.message}`);
                setStatus('Backend Offline');
            }
        }
        checkHealth();
    }, []);

    return `
        <div>
            <h1>Vial MCP Chatbot</h1>
            <div>Status: ${status}</div>
            <div id="console">${output}</div>
            <input type="text" placeholder="Enter API Key" onChange={e => setApiKey(e.target.value)} />
            <input type="text" placeholder="Enter Wallet ID" onChange={e => setWalletId(e.target.value)} />
            <button onClick={authenticate}>Authenticate</button>
            <input type="text" placeholder="Enter Prompt" onChange={e => setPrompt(e.target.value)} />
            <button onClick={sendPrompt}>Send Prompt</button>
            <input type="text" placeholder="Enter Note" onChange={e => setNote(e.target.value)} />
            <button onClick={addNote}>Add Note</button>
            <button onClick={getResources}>Get Latest Resources</button>
            <div id="output">${output}</div>
        </div>
    `;
}
