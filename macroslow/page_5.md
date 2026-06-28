# MAML (Markdown as Medium Language): A Practical Communication Medium for Modern MCP-Based Agentic Harnesses

**Report for Skill.md Integration**  
**Prepared for Agentic AI Developers**  
**Focus: Non-Quantum PCs, MCP Compatibility, Hermes & OpenClaw Harnesses**  
**Version: 1.0 (Adapted from Webxos Concepts, June 2026)**  
**Page 5 of 10**

## 5. MAML-Lite Setup, Execution, and Tooling for Non-Quantum Environments

MAML-Lite is the streamlined, production-ready implementation of MAML optimized for classical computing hardware. It removes all quantum-related features while preserving full MCP compatibility, making it ideal for local development, self-hosted harnesses, and enterprise deployments on standard CPUs.

### Prerequisites and Installation

**System Requirements**:
- Operating System: Linux, macOS, or Windows (with WSL for best results)
- Python 3.8 or higher
- Node.js 18+ (for JavaScript blocks)
- Docker (recommended for sandboxing)
- Git for version control of MAML files

**Core Setup Steps**:

1. **Create Project Directory**:
   ```bash
   mkdir maml-skills && cd maml-skills
   git init
   ```

2. **Install Core Dependencies**:
   ```bash
   pip install pydantic pandas requests pyyaml fastapi uvicorn
   npm install -g node-fetch  # Optional for JS testing
   ```

3. **Simple MCP Server for Testing** (using FastAPI):
   Create `mcp_server.py`:
   ```python
   from fastapi import FastAPI, UploadFile
   import yaml
   import subprocess
   from pathlib import Path

   app = FastAPI()

   @app.post("/execute")
   async def execute_maml(file: UploadFile):
       content = await file.read()
       # Parse front matter and execute code blocks (simplified)
       data = yaml.safe_load(content.split(b'---')[1])
       # Sandbox execution logic here
       result = {"status": "success", "maml_id": data.get("id")}
       return result

   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

4. **Run the Server**:
   ```bash
   python mcp_server.py
   ```

### Executing MAML Files

**Command-Line Execution**:
```bash
curl -X POST -H "Content-Type: text/markdown" \
  --data-binary @my_skill.maml.md \
  http://localhost:8000/execute
```

**Python Execution Helper**:
```python
import yaml
from pathlib import Path

def run_maml(file_path: str):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract front matter
    front_matter = yaml.safe_load(content.split('---')[1])
    print(f"Executing skill: {front_matter.get('intent', 'No intent provided')}")
    
    # In production: parse and run code blocks in sandbox
    # Example: subprocess.run for Python blocks
```

**Docker Sandbox (Recommended for Harnesses)**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "execute_maml.py"]
```

### Tooling and Development Workflow

**Recommended Tools**:
- **Editor Support**: VS Code with Markdown, Python, and YAML extensions. Add MAML language syntax highlighting via custom snippets.
- **Validation**: Custom script using Pydantic and JSON Schema for front matter and schemas.
- **Version Control**: Store `.maml.md` files in Git. Use conventional commits for History synchronization.
- **Testing Framework**: Write companion test scripts that feed sample inputs and assert outputs against schemas.
- **Harnesses Integration**:
  - Hermes: Script to watch MAML directory and update memory on file changes.
  - OpenClaw: Gateway plugin that registers `.maml.md` as discoverable tools.

**Debugging Tips**:
- Add verbose logging in code blocks.
- Use `## History` to record intermediate states.
- Run with resource limits in Docker to simulate production constraints.
- Validate permissions manually before full harness deployment.

### Performance Considerations on Non-Quantum PCs

- Keep individual code blocks lightweight (< 500ms execution on typical hardware).
- Use efficient libraries (pandas for data, not full ML unless GPU available).
- Limit concurrent executions via MCP server queuing.
- Monitor memory usage — MAML files themselves are text-based and compact.

**Example Full Execution Pipeline**:
1. Author skill in `.maml.md`.
2. Validate locally.
3. Submit to MCP test server.
4. Integrate into Hermes/OpenClaw.
5. Iterate using History feedback.

MAML-Lite enables rapid prototyping and deployment of sophisticated agent skills with minimal overhead, serving as the foundation for reliable MCP-based systems on everyday hardware.

(End of Page 5. Continued on Page 6: Skill Definition Templates and Examples for .maml.md.)
