
# 🐍 PygmyClaw v1.3 (Testing)
---

```
▄▖           ▜      
▙▌▌▌▛▌▛▛▌▌▌▛▘▐ ▀▌▌▌▌
▌ ▙▌▙▌▌▌▌▙▌▙▖▐▖█▌▚▚▘
  ▄▌▄▌   ▄▌                                 
```

**PygmyClaw** is a compact Py based agent harness that features a persistent task queue and a modular tool system.

---

## Features

- **Speculative Decoding** – Uses 3 drafters and 1 verifier (four Ollama instances) to produce tokens faster.
- **Persistent Task Queue** – Redis or local JSON file; tasks are processed in the background.
- **Tool‑calling Architecture** – Tools run in a separate subprocess (`pygmyclaw_multitool.py`) for isolation.
- **Interactive REPL** – Chat with the agent and let it decide when to use tools.
- **Command‑line Interface** – Manage instances, queue, and generation with simple subcommands.

```
pygmyclaw/
├── pygmyclaw.py                # Main agent with speculative decoding and queue
├── pygmyclaw_multitool.py      # Tool implementations (echo, sys_info, log_error, etc.)
├── config.json                 # Configuration file (model, ports, queue settings)
```

---

# PygmyClaw v1.3 Update:

The v1.3 update combines the local language model with a set of powerful Python tools. It supports **multi‑instance speculative decoding** for faster generation, a **persistent task queue** (Redis or file‑based), and an integrated **scheduler** for cron‑like jobs. This guide will help you set up and use all the enhanced features.

- **`pygmyclaw.py`** – the main agent that interacts with Ollama, manages instances, queues, and the scheduler.
- **`pygmyclaw_multitool.py`** – a separate process that executes actual tool functions (heartbeat, file I/O, scheduling, etc.). The agent calls it via `subprocess`.

All communication is via JSON. The agent builds a system prompt describing the available tools, and the model can respond with a JSON object to invoke a tool. The result is then fed back to the model for a natural language answer.

---

## Prerequisites

- **Python 3.8+**
- **Ollama** installed and running (with at least one model pulled).  
  - Recommended smaller model: `qwen2.5:0.5b` (small and fast) or any model you prefer.
- Optional but recommended:
  - `psutil` – for detailed heartbeat stats.
  - `dateparser` – for flexible time specifications in scheduler.
  - `redis` – if you want a persistent queue with Redis.
  - `ollama` Python client – for multi‑instance speculative decoding.
  - `crontab` access (if you plan to use system cron integration – not required for the built‑in scheduler).

---

## Installation

1. **Clone or download** the three files:  
   - `pygmyclaw.py`  
   - `pygmyclaw_multitool.py`  
   - `config.json`

2. **Install Python dependencies** (choose according to your needs):

   ```bash
   # Basic
   pip install psutil dateparser

   # For multi‑instance decoding
   pip install ollama

   # For Redis queue
   pip install redis
   ```

3. **Make the scripts executable** (optional on Windows):

   ```bash
   chmod +x pygmyclaw.py pygmyclaw_multitool.py
   ```

4. **Ensure Ollama is running** and the desired model is available:

   ```bash
   ollama serve          # in a separate terminal
   ollama pull qwen2.5:0.5b
   ```

---

## Configuration

Edit `config.json` to suit your environment:

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
    },
    "scheduler": {
        "enabled": true,
        "check_interval": 60
    }
}
```

- **`model`** – the Ollama model to use.
- **`endpoint`** – Ollama API endpoint (usually `http://localhost:11434/api/generate`).
- **`workspace`** – directory where file operations are allowed (all paths are restricted to this directory for security).
- **`debug`** – set to `true` to see detailed logs.
- **`multi_instance`** – enable speculative decoding with 4 Ollama instances on different ports.  
  - `enabled`: `true`/`false`  
  - `ports`: list of 4 ports (must be free).
- **`queue`** – persistent task queue configuration.  
  - `type`: `"redis"` or `"file"` (if Redis is unavailable or `type` omitted, it falls back to a JSON file).
  - `redis_host`, `redis_port`, `queue_name` – only for Redis.
- **`scheduler`** – built‑in scheduler for periodic tasks.  
  - `enabled`: `true`/`false`  
  - `check_interval`: seconds between checks of the jobs file.

---

## Running PygmyClaw

### REPL Mode

The default mode is an interactive REPL (read‑eval‑print loop). Simply run:

```bash
./pygmyclaw.py
```

You’ll see a prompt `>>`. Type your questions or commands. The agent will think, optionally use a tool, and respond.

**Built‑in REPL commands:**

- `/help` – show available tools and system info.
- `/exit` or `/q` – quit.

### Command‑Line Subcommands

PygmyClaw also provides several subcommands for non‑interactive use:

```
usage: pygmyclaw.py [-h] {start,stop,generate,queue,scheduler,repl} ...

PygmyClaw with speculative decoding and scheduler

subcommands:
  {start,stop,generate,queue,scheduler,repl}
    start       Start 4 Ollama instances
    stop        Stop all instances
    generate    Generate text using speculative decoding
    queue       Queue operations
    scheduler   Scheduler operations
    repl        Start interactive REPL (default)
```

Examples:

```bash
# Start multi‑instances in background
./pygmyclaw.py start --background

# Generate text with speculative decoding
./pygmyclaw.py generate "Explain quantum computing" --max-tokens 200

# Add a task to the queue
./pygmyclaw.py queue add "Translate 'hello' to French"

# Start the scheduler (if enabled in config)
./pygmyclaw.py scheduler start

# Stop everything
./pygmyclaw.py stop
```

---

## Using the Tools

All tools are defined in `pygmyclaw_multitool.py`. When the agent starts, it fetches the list and builds a system prompt. You can trigger a tool by asking the agent to do something – it will decide when to output a JSON tool call.

### Listing Tools

In the REPL, type `/help` to see all available tools. You can also ask:

```
>> What tools do you have?
```

The agent will respond with a list (or use the tool internally to list them).

### Heartbeat – System Health

Ask about system status:

```
>> How is the system doing?
>> What's the CPU usage?
>> Show me memory and disk info.
```

The agent will likely call the `heartbeat` tool and then summarise the results.

If `psutil` is installed, you’ll get detailed stats; otherwise a basic platform string.

### File Read / Write

You can read or write files **inside the workspace** (the directory where PygmyClaw is running, or the configured `workspace`).

Examples:

```
>> Read the contents of todo.txt
>> Write "Buy milk" to shopping.txt
>> Append "Eggs" to shopping.txt
```

The agent will call `file_read` or `file_write` accordingly. File paths are resolved relative to the workspace, and any attempt to escape (e.g., `../`) is blocked.

### Scheduler – Cron‑like Jobs

The scheduler allows you to schedule commands to run at a future time. It uses a simple JSON file (`scheduled_jobs.json`) to store jobs. A background thread in the main agent checks this file every `check_interval` seconds and executes due commands.

**Available scheduler tools:**

- `schedule_task(command, time_spec, job_id?)` – schedule a shell command.
- `list_scheduled()` – show all scheduled jobs.
- `remove_scheduled(job_id)` – delete a job.

**Time specification** – you can use natural language like:

- `"in 5 minutes"`
- `"in 2 hours"`
- `"tomorrow at 10am"`

If `dateparser` is installed, many formats are supported. Otherwise, only a very simple `"in X minutes"` pattern works. Recurring specs (e.g., `"every day at 8am"`) are currently ignored (you would need to reschedule after execution).

**Example conversation:**

```
>> Schedule a backup in 10 minutes
>> What jobs are scheduled?
>> Cancel the backup job (by ID)
```

The agent will use the appropriate tools. When a job runs, it executes the command in a shell (like `subprocess.Popen(cmd, shell=True)`). Be mindful of security – the command is whatever you provided.

---

## Multi‑Instance Speculative Decoding

When enabled in `config.json`, PygmyClaw will launch four Ollama instances on different ports (e.g., 11434–11437). Three act as drafters, one as a verifier. This speeds up generation by speculating multiple tokens in parallel.

To use it:

1. Ensure the `ollama` Python client is installed (`pip install ollama`).
2. Set `"enabled": true` in the `multi_instance` section of `config.json`.
3. Start the instances with `./pygmyclaw.py start` (or they will be started automatically when needed if not already running).
4. Use the `generate` subcommand or REPL as usual – the agent will automatically use speculative decoding for long responses.

**Important:** The ports must be free. If other services use those ports, change them in the config. Also, ensure you have enough system resources (RAM/VRAM) to run four model instances simultaneously – for small models like `qwen2.5:0.5b` this is usually fine.

You can stop all instances with `./pygmyclaw.py stop`.

---

## Persistent Task Queue

The queue allows you to submit tasks (prompts) that will be processed asynchronously by a background worker. This is useful for batch jobs or when you don’t want to wait for the result.

**Queue types:**

- **Redis** – if Redis is available and configured, tasks are stored in a Redis list. This survives agent restarts.
- **File** – fallback using a JSON file (`task_queue.json`). Also persistent.

**Commands:**

- Add a task: `./pygmyclaw.py queue add "Your prompt"`
- Start the processor (runs in foreground): `./pygmyclaw.py queue process`
- Check status: `./pygmyclaw.py queue status`

When the processor runs, it takes tasks from the queue, generates a response (using speculative decoding if enabled), and writes the output to a file named `task_<id>.out` in the workspace.

You can also start the processor in the background with the `--background` flag when launching instances.

---

## Troubleshooting

| Problem | Possible Solution |
|--------|-------------------|
| `Cannot reach Ollama` | Ensure Ollama is running (`ollama serve`). Check endpoint in config. |
| `Model '...' not found` | Pull the model: `ollama pull <model>` |
| `ModuleNotFoundError` | Install missing dependencies (psutil, dateparser, redis, ollama). |
| File operations fail with “Path not allowed” | The path must be inside the workspace. Use relative paths (e.g., `notes.txt`, not `/etc/passwd`). |
| Scheduler jobs not running | Check that scheduler is enabled in config and started (`./pygmyclaw.py scheduler start`). Look for error messages in the console. |
| Multi‑instance decoding slow | Ensure all four instances are running (`./pygmyclaw.py start`). If resources are tight, disable multi‑instance. |
| Queue not processing | Start the queue processor with `./pygmyclaw.py queue process` (runs in foreground) or use `--background` when starting. |

---

## License

This project is open source under the MIT License.
