<div align="center">
  <pre style="color: #CD7F32; font-size: 3em; font-weight: bold; line-height: 1.2; margin: 0;">
    
 ██████╗ ██╗   ██╗███████╗████████╗██╗   ██╗ ██████╗██╗      █████╗ ██╗    ██╗
 ██╔══██╗██║   ██║██╔════╝╚══██╔══╝╚██╗ ██╔╝██╔════╝██║     ██╔══██╗██║    ██║
 ██████╔╝██║   ██║███████╗   ██║    ╚████╔╝ ██║     ██║     ███████║██║ █╗ ██║
 ██╔══██╗██║   ██║╚════██║   ██║     ╚██╔╝  ██║     ██║     ██╔══██║██║███╗██║
 ██║  ██║╚██████╔╝███████║   ██║      ██║   ╚██████╗███████╗██║  ██║╚███╔███╔╝
 ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝      ╚═╝    ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ 
    
  </pre>
</div>

# 🦞 RustyClaw 0.6.0 – Local Agent Harness

[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RustyClaw** is a terminal‑based, minimal barebones and OEM local‑only agent harness powered by [Ollama](https://ollama.com/).

It combines a TUI chat interface, file system operations, Git versioning, memory consolidation, and a REST API – all inside a single Rust binary.

---

## ✨ Features

- 🧠 **Persistent memory** – `bio.md` evolves with every conversation.
- 🖥️ **Full‑screen TUI** – built with `ratatui` and `crossterm`.
- 🤖 **Local Ollama** – no data leaves your machine (supports any model).
- 📁 **Sandboxed file ops** – read/write files inside `~/.rustyclaw/data/`.
- 🔐 **Whitelisted shell commands** – `ls`, `cat`, `echo`, `git`, `pwd`.
- 📦 **Git versioning** – every file change is auto‑committed (optional).
- 🧠 **Memory consolidation** – periodic summarisation of conversations into `bio.md`.
- 🌐 **REST API** – `GET /api/bio` to fetch the current `bio.md`.
- 🎨 **Permanent ASCII logo** – RustyClaw branding stays on screen.
- ⚡ **Non‑blocking runtime** – smooth TUI even while background tasks run.

---

## File Structure

```
rustyclaw/
├── src/
│   └── main.rs                 # single‑file application
├── Cargo.toml                  # dependencies
├── start.sh                    # launcher script (build + run)
├── config.yaml                 # optional – auto‑created on first run
├── data/                       # sandboxed file storage (Git repo)
│   └── logs/
│       └── app.log             # JSON log (tracing)
└── ~/.rustyclaw/               # user data directory
    ├── bio.md                  # living agent identity (persistent memory)
    └── data/                   # symlink or actual copy of sandbox
```

> **Note:** `~/.rustyclaw/` is created automatically on first launch.  
> The `data/` folder inside it is initialised as a Git repository if `git` is available.

---

## 🛠️ Installation

### 1. Once you have the files downloaded run this bash command: 
```bash
cd ~/rustyclaw              (The folder you have the files installed)
cargo build --release
./start.sh --rebuild
```

### 2. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                 # start the server
ollama pull qwen2.5:0.5b      # pull a small model (or any you like)
```

### 3. Install Git (optional but recommended)
```bash
sudo apt install git          # Debian/Ubuntu
# or brew install git on macOS
```

Warning: The first build may take a few minutes. Subsequent runs will reuse the cached binary.

---

## Configuration

On first launch, a default `config.yaml` is created in the current directory.  
You can edit it to change behaviour:

```yaml
ollama_url: "http://localhost:11434"
ollama_model: "qwen2.5:0.5b"
api_port: 3030
root_dir: "/home/you/.rustyclaw"
bio_file: "/home/you/.rustyclaw/bio.md"
heartbeat_log: "/home/you/.rustyclaw/data/logs/heartbeat.log"
memory_sync_interval_secs: 3600   # consolidate every hour
max_log_lines: 200
git_auto_commit: true
```

| Field                      | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `ollama_url`               | Ollama API endpoint (default `http://localhost:11434`)                      |
| `ollama_model`             | Model to use for chat and consolidation                                     |
| `api_port`                 | Port for the REST API                                                       |
| `root_dir`                 | Where `bio.md` and `data/` live (default `~/.rustyclaw`)                    |
| `git_auto_commit`          | Automatically commit file writes in the `data/` folder                      |
| `memory_sync_interval_secs`| How often to run automatic memory consolidation                             |

---

## `bio.md` – The Living Agent Memory

`bio.md` is a Markdown file that acts as the agent’s **persistent long‑term memory**.  
It is read on every chat and updated during `/consolidate`. The file is structured into five sections:

### 1. `# BIO.MD – Living Agent Identity`
- Contains the **last updated** timestamp (auto‑refreshed after each chat).

### 2. `## SOUL`
- Core personality, values, constraints, and behavioural rules.  
- Example: *“Stay sandboxed, respect security, be concise and helpful.”*

### 3. `## SKILLS`
- Reusable capabilities and “how‑to” instructions.  
- Example: *“Read/write local files, run whitelisted shell commands.”*

### 4. `## MEMORY`
- Curated long‑term knowledge.  
- During `/consolidate`, the agent summarises recent conversations and appends a new entry here (e.g., `### Summary for 2025-04-02 14:30 …`).

### 5. `## CONTEXT`
- Current runtime state (OS, working directory, active model).

### 6. `## SESSION TREE`
- Pointers or summaries of active conversation branches (currently a placeholder – can be extended).

> **You can edit `bio.md` manually** – the agent will respect your changes in future chats.

---

## Usage – TUI Commands

Launch the TUI with `./start.sh`.  
All commands are typed at the bottom input line and sent with **Enter**.

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/bio` | Display the current `bio.md` content |
| `/consolidate` | Force memory consolidation (summarises recent chats into `## MEMORY`) |
| `/write_file <path> <content>` | Write a file inside `data/` (supports folders) |
| `/read_file <path>` | Read and display a file from `data/` |
| `/model list` | List all available Ollama models |
| `/model select <name>` | Switch to a different model (persists in `config.yaml`) |
| `/list_dir [path]` | List contents of `data/` or a subfolder |
| `/search <query>` | Search for text in all files under `data/` (regex) |
| `/run <command>` | Run a whitelisted shell command (`ls`, `cat`, `echo`, `git`, `pwd`) inside `data/` |
| `/git status` | Show `git status --short` of the `data/` folder |
| `/git log [n]` | Show last `n` commits (default 10) |
| `/git commit <msg>` | Commit all changes in `data/` with a message |
| `/quit` or `/exit` | Exit RustyClaw |

**Any text not starting with `/` is sent as a chat message to the AI.**

---

## REST API

While the TUI is running, a simple HTTP server listens on `http://127.0.0.1:3030`.

- `GET /health` → `{"status":"ok"}`
- `GET /api/bio` → returns the current `bio.md` as JSON:
  ```json
  {"bio": "# BIO.MD – Living Agent Identity\n**Last Updated:** ..."}
  ```

You can use `curl` to fetch the agent’s memory:
```bash
curl http://127.0.0.1:3030/api/bio
```

---

## How Memory Consolidation Works

1. Every chat interaction is logged as a JSON line in `~/.rustyclaw/data/logs/heartbeat.log`.
2. Periodically (default every 3600 seconds), the agent reads the last 20 entries.
3. It sends a summarisation prompt to Ollama.
4. The summary is inserted into the `## MEMORY` section of `bio.md` with a timestamp.
5. The agent’s future chats include the updated `bio.md`, giving it long‑term recall.

You can also trigger consolidation manually with `/consolidate`.

---

## Tool Functions Explained

The core of RustyClaw is the `run_command` dispatcher in `main.rs`.  
Each command is handled in a non‑blocking worker task.

| Function          | Description |
|-------------------|-------------|
| `Chat`            | Sends user message to Ollama together with the full `bio.md` as system prompt. Logs the exchange and updates the timestamp in `bio.md`. |
| `ConsolidateMemory` | Reads heartbeat log, asks Ollama to summarise, inserts summary into `bio.md`. |
| `WriteFile`       | Sanitises path (stays inside `data/`), creates parent directories, writes content, then optionally `git add` + `commit`. |
| `ReadFile`        | Reads a file from `data/` and displays its content in the logs. |
| `ListModels`      | Calls Ollama’s `/api/tags` endpoint and lists available models. |
| `SelectModel`     | Updates `config.yaml` with the new model name. |
| `ListDir`         | Uses `walkdir` to show one‑level directory listing. |
| `SearchFiles`     | Recursively walks `data/` and prints paths of files containing a regex match. |
| `RunCommand`      | Executes a whitelisted command (`ls`, `cat`, `echo`, `git`, `pwd`) inside `data/`. |
| `GitStatus`, `GitLog`, `GitCommit` | Thin wrappers around `git` commands, always run inside `data/`. |
| `Quit`            | Signals the main loop to exit. |

All file operations are **sandboxed** – the `sanitize_path` function ensures no path can escape `~/.rustyclaw/data/`.

---

## Development

To hack on RustyClaw:

The project is a single Rust file (`src/main.rs`). No modules – easy to experiment.

### Adding a new command
1. Add a variant to `enum AppCommand`.
2. Add a branch in `handle_command` (inside `AppState`).
3. Add a matching branch in `run_command` (the dispatcher).
4. Send the command to the worker via `cmd_tx`.

### Changing the UI
The `ui()` function controls layout. The logo is drawn at the top as a `Paragraph`.  
You can adjust colours, add more status lines, or change key bindings.

---

## 📜 License

MIT License
