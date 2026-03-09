# ✏️ PENCILCLAW – Autonomous C++ Coding Agent

```
██████╗ ███████╗███╗   ██╗ ██████╗██╗██╗     ██████╗██╗      █████╗ ██╗    ██╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██║██║    ██╔════╝██║     ██╔══██╗██║    ██║
██████╔╝█████╗  ██╔██╗ ██║██║     ██║██║    ██║     ██║     ███████║██║ █╗ ██║
██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║██║    ██║     ██║     ██╔══██║██║███╗██║
██║     ███████╗██║ ╚████║╚██████╗██║███████╗╚██████╗██████╗██║  ██║╚███╔███╔╝
╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝╚══════╝ ╚═════╝╚═════╝╚═╝  ╚═╝ ╚══╝╚══╝ 
```

**PENCILCLAW** is a C++‑based autonomous coding agent that harnesses your local [Ollama](https://ollama.com/) instance to generate, manage, and execute C++ code. It features a persistent task system, Git integration, and a secure execution environment – all running offline with complete privacy.

---

## Features

- **Code Generation (`/CODE`)** – Generate C++ code for any idea, automatically saved as a `.txt` file.
- **Autonomous Tasks (`/TASK`)** – Start a long‑running coding goal; the agent continues working on it in the background via heartbeat.
- **Task Management** – View status (`/TASK_STATUS`) and stop tasks (`/STOP_TASK`).
- **Code Execution (`/EXECUTE`)** – Compile and run the last generated code block (with safety confirmation).
- **Git Integration** – Every saved file is automatically committed to a local Git repository inside `pencil_data/`.
- **Heartbeat & Keep‑Alive** – Keeps the Ollama model loaded and continues active tasks periodically.
- **Secure by Design** – Command injection prevented, path sanitisation, explicit confirmation before running AI‑generated code.
- **Natural Language Interface** – Commands like *"write code for a fibonacci function"* are understood.

---

## Project Structure

```
/home/kali/pencilclaw/
├── pencilclaw.cpp          # Main program source
├── pencil_utils.hpp        # Workspace utilities
├── pencilclaw              # Compiled executable
**└── pencil_data/            # Created automatically on first run**
    ├── session.log         # Full interaction log
    ├── .git/               # Local Git repository (if initialised)
    ├── tasks/              # Autonomous task folders
    │   └── 20260309_123456_build_calculator/
    │       ├── description.txt
    │       ├── log.txt
    │       ├── iteration_1.txt
    │       └── ...
    └── [code files].txt    # Files saved via /CODE or natural language
```

---

## Requirements

- **Compiler** with C++17 support (g++ 7+ or clang 5+)
- **libcurl** development libraries
- **nlohmann/json** (header‑only JSON library)
- **Ollama** installed and running
- A model pulled in Ollama (default: `qwen2.5:0.5b` – configurable via environment variable `OLLAMA_MODEL`)

*Note: PENCILCLAW uses POSIX system calls (`fork`, `pipe`, `execvp`). It runs on Linux, macOS, and Windows Subsystem for Linux (WSL).*

---

## Installation

### 1. Install System Dependencies
```bash
sudo apt update
sudo apt install -y build-essential libcurl4-openssl-dev
```

### 2. Install nlohmann/json
The library is header‑only; simply download `json.hpp` and place it in your include path, or install via package manager:
```bash
sudo apt install -y nlohmann-json3-dev
```

### 3. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # start the service
ollama pull qwen2.5:0.5b   # or your preferred model
```

Set Model (Optional)

Override the default model by setting the environment variable:
```bash
export OLLAMA_MODEL="llama3.2:latest"
```
### 4. cd
```bash
cd ~/pencilclaw/      -The folder you have the files installed
```

### 5. Compile PENCILCLAW
```bash
g++ -std=c++17 -o pencilclaw pencilclaw.cpp -lcurl
```
If `json.hpp` is in a non‑standard location, add the appropriate `-I` flag.

---

## Usage

Start the program:
```bash
./pencilclaw
```

You will see the `>` prompt. Commands are case‑sensitive and start with `/`. Any line not starting with `/` is treated as natural language and passed to Ollama.

### Available Commands

| Command               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `/HELP`               | Show this help message.                                                     |
| `/CODE <idea>`        | Generate C++ code for the given idea; saved as `<sanitized_idea>.txt`.      |
| `/TASK <description>` | Start a new autonomous coding task (creates a timestamped folder).         |
| `/TASK_STATUS`        | Show the current active task, its folder, and iteration count.             |
| `/STOP_TASK`          | Clear the active task (does not delete existing task files).               |
| `/EXECUTE`            | Compile and run the first C++ code block from the last AI output.          |
| `/FILES`              | List all saved `.txt` files and task folders.                              |
| `/DEBUG`              | Toggle verbose debug output (shows JSON requests/responses).               |
| `/EXIT`               | Quit the program.                                                           |

### Natural Language Examples

- `write code for a fibonacci function`
- `start a task to build a calculator`
- `save it as mycode.txt` (after code generation)

---

## Git Integration

PENCILCLAW automatically initialises a Git repository inside `pencil_data/` on first run. Every file saved via `/CODE` or task iteration is committed with a descriptive message. The repository is configured with a local identity (`pencilclaw@local` / `PencilClaw`) so commits work even without global Git configuration.

If you prefer not to use Git, simply remove the `.git` folder from `pencil_data/` – PENCILCLAW will detect its absence and skip all Git operations.

---

## Security Notes

- **Code execution is potentially dangerous.** PENCILCLAW always shows the code and requires you to type `yes` before running it.
- **Path traversal is prevented** – filenames are sanitised, and all writes are confined to `pencil_data/`.
- **No shell commands are used** – all external commands (`git`, `g++`) are invoked via `fork`+`execvp` with argument vectors, eliminating command injection risks.

---

## Configuration

| Setting                | Method                                                                 |
|------------------------|------------------------------------------------------------------------|
| Ollama model           | Environment variable `OLLAMA_MODEL` (default: `qwen2.5:0.5b`)         |
| Workspace directory    | Environment variable `PENCIL_DATA` (default: `./pencil_data/`)        |
| Heartbeat interval     | Edit `HEARTBEAT_INTERVAL` in source (default 120 seconds)             |
| Keep‑alive interval    | Edit `KEEP_ALIVE_INTERVAL` in source (default 120 seconds)            |

---

## Troubleshooting

| Problem                          | Solution                                                       |
|----------------------------------|----------------------------------------------------------------|
| `json.hpp: No such file or directory` | Install nlohmann/json or add the correct `-I` flag.          |
| `curl failed: Couldn't connect to server` | Ensure Ollama is running (`ollama serve`) and the URL `http://localhost:11434` is accessible. |
| Model not found                  | Run `ollama pull <model_name>` (e.g., `qwen2.5:0.5b`).        |
| Git commit fails                 | The repository already has a local identity; this should not happen. If it does, run `git config` manually in `pencil_data/`. |
| Compilation errors (C++17)       | Use a compiler that supports `-std=c++17` (g++ 7+ or clang 5+). |

---

## License

This project is released under the MIT License. Built with C++ and Ollama.
