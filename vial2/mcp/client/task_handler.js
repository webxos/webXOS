export class TaskHandler {
    constructor() {
        this.tasks = [];
    }

    async addTask(taskData) {
        try {
            const response = await fetch('/vial/task/manage', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: taskData })
            });
            const data = await response.json();
            if (data.result) {
                this.tasks.push(data.result.data);
                this.renderTasks();
                return true;
            }
            return false;
        } catch (e) {
            console.error(`Task addition failed: ${e.message}`);
            return false;
        }
    }

    renderTasks() {
        const consoleDiv = document.getElementById('console');
        if (consoleDiv) {
            consoleDiv.innerHTML += `<p>Tasks: ${JSON.stringify(this.tasks)}</p>`;
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        }
    }
}

export const taskHandler = new TaskHandler();

# xAI Artifact Tags: #vial2 #mcp #client #task #handler #neon_mcp
