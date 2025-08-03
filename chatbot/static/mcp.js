class MCPController {
    constructor() {
        this.eventBus = new EventTarget();
        this.agents = {
            agent1: { name: 'Agent1', tasks: ['process', 'fetch', 'summarize', 'visualize', 'filter', 'cluster', 'respond', 'validate', 'suggest', 'output'], color: 'agent1-color' },
            agent2: { name: 'Agent2', tasks: ['process', 'fetch', 'summarize', 'visualize', 'filter', 'cluster', 'respond', 'validate', 'suggest', 'output'], color: 'agent2-color' },
            agent3: { name: 'Agent3', tasks: ['process', 'fetch', 'summarize', 'visualize', 'filter', 'cluster', 'respond', 'validate', 'suggest', 'output'], color: 'agent3-color' },
            agent4: { name: 'Agent4', tasks: ['process', 'fetch', 'summarize', 'visualize', 'filter', 'cluster', 'respond', 'validate', 'suggest', 'output'], color: 'agent4-color' }
        };
        this.taskQueue = [];
        this.sessionData = JSON.parse(sessionStorage.getItem('mcpData')) || { apiCalls: 0, errors: 0 };
        this.learningEnabled = true;
        this.activeAgents = 0;
        this.apiCalls = this.sessionData.apiCalls || 0;
        this.errorCount = this.sessionData.errors || 0;
        this.apiOutputLimit = 100;
        this.updateStats();
    }

    initialize() {
        // Initialization deferred until /mcp command
    }

    async loadAgents() {
        for (const agent of Object.keys(this.agents)) {
            try {
                await loadScript(`/chatbot/static/${agent}.js`);
                console.log(`${agent} loaded`);
            } catch (error) {
                console.error(`Failed to load ${agent}:`, error);
                updateStatus(`Error: Failed to load ${agent}.`, true, error.message);
                this.incrementErrorCount();
            }
        }
    }

    handleAgentOutput(event) {
        const { agent, task, output } = event.detail;
        if (this.learningEnabled) {
            this.sessionData[agent] = this.sessionData[agent] || [];
            this.sessionData[agent].push({ task, output, timestamp: Date.now() });
            sessionStorage.setItem('mcpData', JSON.stringify(this.sessionData));
        }
        this.taskQueue.shift();
        this.activeAgents = Math.max(0, this.activeAgents - 1);
        this.updateStats();
        this.processNextTask();
        updateNeuralCanvas(agent, task, output);
        const messages = document.getElementById('messages');
        messages.innerHTML += `<p class="${this.agents[agent].color}"><b>${agent.toUpperCase()}:</b> ${sanitizeInput(JSON.stringify(output))}</p>`;
        messages.scrollTop = messages.scrollHeight;
        if (task === 'output' && !document.getElementById('mcpPopup').classList.contains('hidden')) {
            displayApiOutput(agent, output);
        }
    }

    dispatchTask(agent, task, data) {
        if (agent === 'all') {
            Object.keys(this.agents).forEach(a => {
                this.taskQueue.push({ agent: a, task, data });
                this.activeAgents = Math.min(Object.keys(this.agents).length, this.activeAgents + 1);
            });
        } else {
            this.taskQueue.push({ agent, task, data });
            this.activeAgents = Math.min(Object.keys(this.agents).length, this.activeAgents + 1);
        }
        this.updateStats();
        this.processNextTask();
    }

    processNextTask() {
        if (this.taskQueue.length === 0 || document.getElementById('mcpPopup').classList.contains('hidden')) return;
        const { agent, task, data } = this.taskQueue[0];
        const event = new CustomEvent(`${agent}_task`, { detail: { task, data } });
        this.eventBus.dispatchEvent(event);
    }

    updateStats() {
        if (!document.getElementById('mcpPopup').classList.contains('hidden')) {
            document.getElementById('activeAgents').textContent = this.activeAgents;
            document.getElementById('taskQueue').textContent = this.taskQueue.length;
            document.getElementById('apiCalls').textContent = this.apiCalls;
            document.getElementById('errorCount').textContent = this.errorCount;
        }
    }

    incrementErrorCount() {
        this.errorCount++;
        this.sessionData.errors = this.errorCount;
        sessionStorage.setItem('mcpData', JSON.stringify(this.sessionData));
        this.updateStats();
    }

    async fetchFromAgentAPI(agent, query) {
        if (this.apiCalls >= this.apiOutputLimit) {
            this.incrementErrorCount();
            return { error: `API call limit (${this.apiOutputLimit}) reached for ${agent}` };
        }
        const config = JSON.parse(sessionStorage.getItem(`${agent}Config`) || '{}');
        if (!config.apiUrl) {
            this.incrementErrorCount();
            return { error: `No API configured for ${agent}` };
        }
        try {
            const headers = {};
            if (config.apiKey) headers['Authorization'] = `Bearer ${sanitizeInput(config.apiKey)}`;
            if (config.credentials) headers['X-Credentials'] = sanitizeInput(config.credentials);
            const response = await fetch(`${sanitizeInput(config.apiUrl)}?query=${encodeURIComponent(sanitizeInput(query))}`, { headers });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.apiCalls++;
            this.sessionData.apiCalls = this.apiCalls;
            sessionStorage.setItem('mcpData', JSON.stringify(this.sessionData));
            this.updateStats();
            const data = await response.json();
            return { data: sanitizeInput(JSON.stringify(data)) };
        } catch (error) {
            this.incrementErrorCount();
            return { error: `API fetch failed for ${agent}: ${error.message}` };
        }
    }

    async outputAPI(agent, data) {
        const output = {
            agent,
            results: data.results || [],
            timestamp: new Date().toISOString()
        };
        this.sessionData[agent] = this.sessionData[agent] || [];
        this.sessionData[agent].push({ task: 'output', output, timestamp: Date.now() });
        sessionStorage.setItem('mcpData', JSON.stringify(this.sessionData));
        this.apiCalls++;
        this.sessionData.apiCalls = this.apiCalls;
        this.updateStats();
        return output;
    }

    handleMCPCommand(command, args) {
        const messages = document.getElementById('messages');
        switch (command) {
            case 'status':
                if (document.getElementById('mcpPopup').classList.contains('hidden')) {
                    return `<p><b>Error:</b> Open /mcp popup first to view status.</p>`;
                }
                const status = Object.values(this.agents).map(a => `${a.name}: ${a.currentTask || 'Idle'}`).join(', ');
                return `<p><b>MCP Status:</b> ${status}</p>`;
            case 'assign':
                const [agent, task] = args;
                if (!this.agents[agent] && agent !== 'all' || !this.agents[agent]?.tasks.includes(task)) {
                    this.incrementErrorCount();
                    return `<p><b>Error:</b> Invalid agent or task.</p>`;
                }
                this.dispatchTask(agent, task, args.slice(2).join(' '));
                return `<p><b>Success:</b> Assigned ${task} to ${agent}.</p>`;
            case 'chain':
                const tasks = args.join(' ').split(' > ').map(t => t.trim());
                tasks.forEach((t, i) => {
                    const [agent, task] = t.split(':');
                    if ((this.agents[agent] || agent === 'all') && this.agents[agent]?.tasks.includes(task)) {
                        this.dispatchTask(agent, task, i === 0 ? args.join(' ') : null);
                    }
                });
                return `<p><b>Success:</b> Task chain started.</p>`;
            case 'learn':
                this.learningEnabled = args[0] === 'on';
                return `<p><b>Success:</b> Learning ${this.learningEnabled ? 'enabled' : 'disabled'}.</p>`;
            case 'reset':
                this.taskQueue = [];
                this.activeAgents = 0;
                this.errorCount = 0;
                this.apiCalls = 0;
                sessionStorage.removeItem('mcpData');
                this.sessionData = { apiCalls: 0, errors: 0 };
                document.getElementById('agent1Output').innerHTML = '';
                document.getElementById('agent2Output').innerHTML = '';
                document.getElementById('agent3Output').innerHTML = '';
                document.getElementById('agent4Output').innerHTML = '';
                return `<p><b>Success:</b> MCP reset.</p>`;
            case 'visualize':
                if (document.getElementById('mcpPopup').classList.contains('hidden')) {
                    return `<p><b>Error:</b> Open /mcp popup first to visualize.</p>`;
                }
                this.dispatchTask('all', 'visualize', args[0] || 'workflow');
                return `<p><b>Success:</b> Visualization triggered.</p>`;
            case 'priority':
                const [agent, level] = args;
                if (this.agents[agent]) {
                    this.taskQueue.sort((a, b) => a.agent === agent && level === 'high' ? -1 : 1);
                    return `<p><b>Success:</b> Priority set for ${agent}.</p>`;
                }
                this.incrementErrorCount();
                return `<p><b>Error:</b> Invalid agent.</p>`;
            case 'output':
                if (document.getElementById('mcpPopup').classList.contains('hidden')) {
                    return `<p><b>Error:</b> Open /mcp popup first to generate output.</p>`;
                }
                this.dispatchTask('all', 'output', args.join(' '));
                return `<p><b>Success:</b> API output generation triggered.</p>`;
            default:
                this.incrementErrorCount();
                return `<p><b>Error:</b> Unknown MCP command. Try /mcp status, assign, chain, learn, reset, visualize, filter, cluster, respond, suggest, output.</p>`;
        }
    }
}

const mcp = new MCPController();
window.addEventListener('load', () => {
    // Load agents only when needed, deferred to /mcp command
});
