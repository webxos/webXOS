(Under Development)
```
  ███████╗████████╗ █████╗  ██████╗██╗  ██╗
  ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
  ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
  ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
  ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

**Embed‑Driven Repository Architect**  
*Local, offline, zero‑API‑key code templating & versioning, powered by Ollama embeddings*
</div>

---

STACK is a **single‑file Python CLI** that builds, templates, and version‑controls projects using **vector embeddings**.  
It uses the **nomic‑embed‑text** model (running locally via [Ollama](https://ollama.com)) to semantically match natural language descriptions with pre‑built code templates.  
Everything happens **offline**, **no API keys**, **no cloud** — your code never leaves your machine.

### Features

- **Zero‑dependency bootstrapping** – auto‑creates a Python virtual environment on first run.
- **Semantic template injection** – describe what you want (`web server`, `login form`) and STACK injects the best matching files.
- **Extensible template system** – hot‑reloads templates from a folder, supports JSON imports.
- **Vector memory** – remembers your past actions and retrieves them for contextual suggestions.
- **Git‑first** – every injection happens in a feature branch, merged automatically.
- **Safe by design** – path traversal protection, atomic manifest writes, and graceful error handling.

---

## Requirements

- Python 3.10+
- Ollama installed and running (`ollama serve`).
- `git` available on your `PATH`.
- Ollama model nomic-embed-text pulled on to your system

STACK installs its own dependencies into an isolated virtual environment – you never need to run `pip install` yourself.

---

## Quick Start

```bash
# Place the stack.py file in a folder on your system
cd ~/stack/             (The folder you have the file in)
python3 stack.py
```

The first launch will:
1. Create a virtual environment (`.stack_venv/`).
2. Install `ollama` and other dependencies inside it.
3. Pull the `nomic-embed-text` model if missing.
4. Drop you into the interactive STACK shell.

---

## Commands

Inside the STACK shell (after the banner), type `/help` to see the full list:

| Command | What it does |
|--------|-------------|
| `/build <name>` | Create a new Git repository workspace |
| `/add <description>` | Find the best matching template and inject it into the workspace |
| `/import <file.json>` | Import external template bundles |
| `/list` | Show all available templates |
| `/status` | Display system status (workspace, Ollama, templates) |
| `/help` | Show this help |
| `/quit` | Exit STACK |

**Example workflow:**
```
STACK > /build myproject
STACK > /add python web server with fastapi
STACK > /list
STACK > /status
STACK > /quit
```

After `/add`, your workspace will contain the injected files, committed on a feature branch and merged to `main`.

---

## 📂 Template Management

Templates can be added in two ways:

### 1. Folder hot‑reload
Place folders inside `stack_system/templates/`.  
Each folder should contain:
- Code files (any structure)
- An optional `description.txt` (first line becomes the embedding description)

STACK will automatically detect changes every 2 seconds and regenerate the manifest.

### 2. JSON import
Create a `.json` file like:
```json
{
  "template_name": {
    "description": "Natural language description for embedding",
    "files": {
      "path/relative/to/workspace": "file content here"
    }
  }
}
```
Then run `/import path/to/template.json` inside STACK.

---

## 📁 System Files

| Path | Purpose |
|------|--------|
| `./stack_system/` | Root directory for all STACK data |
| `./stack_system/current_workspace/` | Active Git workspace (changes after `/build`) |
| `./stack_system/templates/` | Hot‑reloadable template folder |
| `./stack_system/templates/manifest.json` | Auto‑generated template index (with embeddings) |
| `./stack_system/.stack_memory.json` | Vector memory store |
| `.stack_venv/` | Virtual environment (auto‑created) |

---

## License

MIT 

