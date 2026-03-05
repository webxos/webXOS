
# 🐍 PygmyClaw v1.0 (Testing)
---

```
    ____                                   __              
   / __ \__  ______ _____ ___  __  _______/ /___ __      __
  / /_/ / / / / __ `/ __ `__ \/ / / / ___/ / __ `/ | /| / /
 / ____/ /_/ / /_/ / / / / / / /_/ / /__/ / /_/ /| |/ |/ / 
/_/    \__, /\__, /_/ /_/ /_/\__, /\___/_/\__,_/ |__/|__/  
      /____//____/          /____/                                                
```

**PygmyClaw** is a lightweight Py based openclaw clone that features a persistent task queue (Redis or file‑based) and a modular tool system implemented in a separate Python script.

---

## Features

- **Speculative Decoding** – Uses 3 drafters and 1 verifier (four Ollama instances) to produce tokens faster.
- **Persistent Task Queue** – Redis or local JSON file; tasks are processed in the background.
- **Tool‑calling Architecture** – Tools run in a separate subprocess (`pygmyclaw_multitool.py`) for isolation.
- **Interactive REPL** – Chat with the agent and let it decide when to use tools.
- **Command‑line Interface** – Manage instances, queue, and generation with simple subcommands.

pygmyclaw/
├── pygmyclaw.py                # Main agent with speculative decoding and queue
├── pygmyclaw_multitool.py      # Tool implementations (echo, sys_info, log_error, etc.)
├── config.json                 # Configuration file (model, ports, queue settings)
├── README.md                    # Documentation (this file)
├── error_log.json               # Created at runtime when errors occur (optional)
└── task_queue.json              # Created when using file-based queue (optional)

---

## Requirements

- Python 3.8 or newer
- [Ollama](https://ollama.com/) installed and available in your `PATH`
- Recommended: Redis (for queue persistence) – optional, falls back to file queue
- Python packages:
  ```bash
  pip install ollama redis
  ```

---

## Quick Start

1. **Pull a model** (e.g., `qwen2.5:0.5b`):
   ```bash
   ollama pull qwen2.5:0.5b
   ```

2. **Start the four Ollama instances** (required for speculative decoding):
   ```bash
   python3 pygmyclaw.py start
   ```
   This launches instances on ports 11434–11437. Keep this terminal open.

3. **Run the REPL** (in another terminal):
   ```bash
   python3 pygmyclaw.py repl
   ```
   Type `/help` to see available commands and tools.

---

## Configuration

Edit `config.json` in the same directory as the script:

```json
{
    "model": "qwen2.5:0.5b",
    "endpoint": "http://localhost:11434/api/generate",
    "workspace": ".",
    "debug": false,
    "multi_instance": {
        "enabled": true,
        "ports": [11434, 11435, 11436, 11437]
    },
    "queue": {
        "type": "redis",
        "redis_host": "localhost",
        "redis_port": 6379,
        "queue_name": "grok_tasks"
    }
}
```

- `model` – Ollama model name.
- `endpoint` – Ollama generate endpoint (usually `http://localhost:11434/api/generate`).
- `workspace` – Directory for logs and task files. Use `"."` for the current directory.
- `debug` – Set to `true` for verbose logging.
- `multi_instance` – Enable/disable speculative decoding and define the ports.
- `queue` – Redis settings; if Redis is unavailable, the queue falls back to a JSON file automatically.

---

## Usage

### Start / Stop Instances

```bash
python3 pygmyclaw.py start                # run in foreground (Ctrl+C to stop)
python3 pygmyclaw.py start --background   # start and detach (queue processor runs in bg)
python3 pygmyclaw.py stop                 # stop all instances
```

### Generate Text (Single‑shot)

```bash
python3 pygmyclaw.py generate "Your prompt" --max-tokens 150
```

### Task Queue

```bash
python3 pygmyclaw.py queue add "Your prompt"      # add a task
python3 pygmyclaw.py queue process                 # process queue (foreground)
python3 pygmyclaw.py queue status                  # show queue length
```

### Interactive REPL

```bash
python3 pygmyclaw.py repl
```

Within the REPL, you can type any query. If the agent decides to use a tool, it will output a JSON object like:

```json
{"tool": "echo", "parameters": {"text": "Hello"}}
```

The tool is executed and the result is fed back to the model for a final response.

---

## Tool Development

Tools are defined in `pygmyclaw_multitool.py`. Each tool must:

1. Have an entry in the `TOOLS` dictionary with keys:
   - `name` – tool name (used by the agent)
   - `description` – short description for the system prompt
   - `parameters` – a dict describing expected parameters (for documentation)
   - `func` – the name of the function to call (e.g., `"do_echo"`)
2. Implement the corresponding function (e.g., `do_echo`) that returns a JSON‑serializable dictionary.
3. The function receives parameters as keyword arguments from the agent.

Example tool (`echo`):

```python
def do_echo(text):
    return {"echo": text}
```

The tool runs in a separate process, so it is isolated from the main agent.

---

## Troubleshooting

- **Ollama not reachable** – Ensure `ollama serve` is running, or that you have started the instances with `pygmyclaw.py start`.
- **Model not found** – Pull the model manually: `ollama pull <model>`.
- **Redis errors** – If Redis is not installed or not running, the queue automatically falls back to a JSON file (`task_queue.json` in the workspace). No action needed.
- **Speculative decoding fails** – If any drafter fails, the system falls back to single‑instance generation. Check that all four ports are free and that Ollama is installed correctly.
- **Tool calls fail** – Verify that `pygmyclaw_multitool.py` is in the same directory and contains the tool implementations. Check the error log (`error_log.json`) for details.

---

## License

This project is open source under the MIT License.
