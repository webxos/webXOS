class Agent {
    constructor(eventBus, name) {
        this.eventBus = eventBus;
        this.name = name;
        this.eventBus.addEventListener(`${name}_task`, (e) => this.handleTask(e.detail));
    }

    async handleTask({ task, data }) {
        let output;
        switch (task) {
            case 'process':
                output = await this.process(data);
                break;
            case 'fetch':
                output = await mcp.fetchFromAgentAPI(this.name, data);
                break;
            case 'summarize':
                output = this.summarize(data);
                break;
            case 'visualize':
                output = this.visualize(data);
                break;
            case 'filter':
                output = this.filter(data);
                break;
            case 'cluster':
                output = this.cluster(data);
                break;
            case 'respond':
                output = this.respond(data);
                break;
            case 'validate':
                output = this.validate(data);
                break;
            case 'suggest':
                output = this.suggest(data);
                break;
            case 'output':
                output = await this.output(data);
                break;
            default:
                output = { error: `Unknown task: ${task}` };
                mcp.incrementErrorCount();
        }
        this.eventBus.dispatchEvent(new CustomEvent(`${this.name}_output`, { detail: { agent: this.name, task, output } }));
    }

    async process(data) {
        const { query, results } = typeof data === 'string' ? JSON.parse(data) : data;
        const apiData = await mcp.fetchFromAgentAPI(this.name, query);
        if (apiData.error) return apiData;
        const keywords = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
        const suggestions = (cachedContent['site_index'] || []).flatMap(item => item.text?.keywords || [])
            .filter(k => k.includes(query.toLowerCase())).slice(0, 5);
        return { keywords, suggestions, apiData, results };
    }

    summarize(data) {
        const text = typeof data === 'string' ? data : JSON.stringify(data);
        return { summary: sanitizeInput(text.slice(0, 100)) + '...' };
    }

    visualize(type) {
        updateNeuralCanvas(this.name, 'visualize', type);
        return { status: `Visualizing ${type}` };
    }

    filter(data) {
        const { results, filter } = typeof data === 'string' ? JSON.parse(data) : data;
        if (!filter || !results) {
            mcp.incrementErrorCount();
            return { error: 'Missing filter or results' };
        }
        const [key, value] = filter.split(':');
        return { filtered: results.filter(item => item[key]?.toLowerCase() === value.toLowerCase()) };
    }

    cluster(data) {
        const { results } = typeof data === 'string' ? JSON.parse(data) : data;
        if (!results) {
            mcp.incrementErrorCount();
            return { error: 'Missing results' };
        }
        const clusters = {};
        results.forEach(item => {
            const key = item.text?.keywords?.[0] || 'other';
            clusters[key] = clusters[key] || [];
            clusters[key].push(item);
        });
        return { clusters };
    }

    respond(data) {
        const { query } = typeof data === 'string' ? JSON.parse(data) : data;
        return { response: `Response to "${sanitizeInput(query)}": Data processed by ${this.name}.` };
    }

    validate(data) {
        const query = typeof data === 'string' ? data : data.query;
        return { valid: query.startsWith('/') ? query.split(' ')[0].length > 1 : true };
    }

    suggest(data) {
        const query = typeof data === 'string' ? data : data.query;
        const suggestions = (cachedContent['site_index'] || []).flatMap(item => item.text?.keywords || [])
            .filter(k => k.toLowerCase().includes(query.toLowerCase())).slice(0, 5);
        return { suggestions };
    }

    async output(data) {
        const { results } = typeof data === 'string' ? JSON.parse(data) : data;
        const agentIndex = Object.keys(this.agents).indexOf(this.name);
        const startIndex = agentIndex * Math.ceil(results.length / 4);
        const endIndex = startIndex + Math.ceil(results.length / 4);
        const agentResults = results.slice(startIndex, endIndex);
        return await mcp.outputAPI(this.name, { results: agentResults });
    }
}

new Agent(mcp.eventBus, 'agent2');
