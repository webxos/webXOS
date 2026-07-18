
# Under Development
```

    ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███████╗███████╗██╗      
   ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██╔════╝██╔════╝██║      
   ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║█████╗  █████╗  ██║      
   ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██╔══██║██╔══╝  ██╔══╝  ╚═╝      
   ╚██████╔╝██║     ███████╗██║ ╚████║╚██████╗██║  ██║███████╗██║     ██╗      
    ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝
```

OPENCHEF is a framework for orchestrating collaborative, sandboxed AI agents. It combines a modern web dashboard, a rich CLI, and a full‑featured backend that manages agent lifecycles, skill execution, long‑term memory, evolutionary brainstorming, autonomous research, dataset extraction, and template‑based project scaffolding — all powered by local Ollama models. OPENCHEF re‑imagines multi‑agent systems as a **skill‑first** platform. Instead of treating agents as black boxes, the system exposes a rich ecosystem of reusable skills, memory, and collaboration tools. Agents are lightweight, sandboxed processes that can be spawned, updated, and retired on the fly. They communicate through channels, execute MAML‑defined skills, reflect on their own outputs, and participate in global workspace context. The system is built for local execution with Ollama, ensuring full data privacy and zero vendor lock‑in. It runs equally well on a laptop or a headless server.

---

## Features

-   **Multi‑Agent Orchestration** – spawn, update, and remove agents; agents can be assigned to specific channels and given custom system prompts.
-   **Skill Engine** – load Python, Bash, C, and MAML (Markdown + YAML frontmatter) skills; skills are executed in isolated workspaces with resource limits.
-   **Reverse Skill Generation** – describe a skill in plain English and let the LLM generate a working Python skill script.
-   **Siphon Research Engine** – perform multi‑source research (DuckDuckGo, SerpApi, web scraping); extract facts, generate summaries, and post results to the siphon channel.
-   **Ralph Evolutionary Brainstorming** – evolve project specifications across generations; uses mutation, evaluation, and convergence detection to refine ideas.
-   **Abstract Planner** – generate structured, step‑by‑step plans for any query and post them as messages.
-   **Dataset Engine** – define extraction schemas in Markdown; scrape URLs, extract structured fields via LLM, and store results in JSON, JSONL, or vector DB (ChromaDB).
-   **Stack Templates** – manage reusable project templates; build repositories from templates, add templates by description, and bulk import.
-   **Moderator & Code Repos** – automatically extract code blocks from agent messages, lint them (ruff, flake8, biome, etc.), and commit them to per‑thread Git repositories.
-   **DecentMem Memory** – agents learn from experience; high‑score trajectories are stored in an `e‑pool`, low‑score ones in an `x‑pool`; retrieval uses cosine similarity over embeddings.
-   **J‑Space Workspace Context** – the system maintains a dynamic set of key concepts from recent conversations; this context is injected into agent prompts, fostering collaborative awareness.
-   **Cron Engine** – schedule periodic webhook calls using cron expressions (requires `croniter`).
-   **Web Dashboard** – built‑in Web UI with chat, agent management, skill browser, Ralph monitoring, and Siphon history.
-   **Full CLI Mode** – run headless with all commands available via terminal.
-   **Circuit‑Broken Ollama Client** – resilient against transient failures; automatically retries and backs off.
-   **Built‑in Metrics & Logging** – real‑time CPU, memory, token rate; rotating file logs; error log with timestamps.
-   **Automatic Database Backups** – scheduled backups with retention management.

---

## Architecture

OPENCHEF is structured as a set of cooperating modules, all orchestrated by the `DashboardServer` (or `CLIMode` for headless operation).

```
┌─────────────────────────────────────────────────────────────────┐
│                     WebSocket / CLI / HTTP                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       DashboardServer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ AgentLoop   │  │ SkillEngine  │  │ Moderator               │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ ResearchEng │  │ DatasetEng   │  │ CronEngine              │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ RalphEngine │  │ AbstractEng  │  │ StackEngine             │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ DecentMem   │  │ BioManager   │  │ OllamaClient            │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                     Storage Layer                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ SQLite (DB) │  │ File System  │  │ ChromaDB (optional)     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

-   **AgentLoop** – core message processing; dispatches to agents, handles skills, reflection, and channel personality.
-   **SkillEngine** – loads and executes skills; supports Python, Bash, C, and MAML.
-   **Moderator** – code extraction, linting, and Git commit to per‑thread repos.
-   **ResearchEngine** – Siphon research: search, scrape, summarise.
-   **DatasetEngine** – structured extraction from web pages.
-   **CronEngine** – periodic job scheduling.
-   **RalphEngine** – evolutionary brainstorming.
-   **AbstractEngine** – plan generation.
-   **StackEngine** – template management.
-   **DecentMem** – agent memory with e‑pool/x‑pool and embedding‑based retrieval.
-   **BioManager** – manages bio.md and soul.md; updates heartbeat summaries.
-   **OllamaClient** – async client with circuit breaker, rate limiting, and token tracking.

---

## Requirements

-   **Python** 3.9 or later
-   **Ollama** running locally (or reachable) with at least one model pulled (e.g., `qwen2.5:0.5b`).
-   **Dependencies** (automatically installed):
    -   `aiohttp`, `aiosqlite`, `pydantic`, `pydantic-settings`
    -   Optional: `croniter`, `PyYAML`, `beautifulsoup4`, `GitPython`, `chromadb`, `psutil`, `numpy`, `tiktoken`

---

## Installation (make sure all needed Requirements are installed)

1.  Place the openchef.py file into an ~/openchef/ folder on your system.

2.  Run:
```bash
cd ~/openchef/ (The Folder the openchef.py file is in) 
python3 openchef.py
```
---

## Configuration

All configuration is driven by **Pydantic Settings**. The system reads from a `.env` file and environment variables with the prefix `OPENCHEF_`. Default Ollama models can be swapped out for use.

### Core settings

| Variable                    | Default                         | Description                                     |
|-----------------------------|---------------------------------|-------------------------------------------------|
| `OPENCHEF_OLLAMA_URL`       | `http://localhost:11434`        | Ollama endpoint                                 |
| `OPENCHEF_DEFAULT_GENERATE_MODEL` | `qwen2.5:0.5b`            | Default model for text generation               |
| `OPENCHEF_DEFAULT_EMBED_MODEL`    | `nomic-embed-text`          | Default model for embeddings                    |
| `OPENCHEF_HOST`             | `0.0.0.0`                       | Web server host                                 |
| `OPENCHEF_PORT`             | `3721`                          | Web server port                                 |
| `OPENCHEF_MAX_ITERATIONS`   | `12`                            | Max turns per agent                             |
| `OPENCHEF_TOKEN_BUDGET`     | `8192`                          | Context window token budget                     |
| `OPENCHEF_MAX_GENERATIONS`  | `30`                            | Ralph max generations                           |
| `OPENCHEF_CONVERGENCE_THRESHOLD` | `0.95`                    | Ralph convergence threshold                     |
| `OPENCHEF_POPULATION_SIZE`  | `3`                             | Ralph population size                           |
| `OPENCHEF_ENABLE_DUCKDUCKGO`| `true`                          | Enable DuckDuckGo search in Siphon              |
| `OPENCHEF_SERPAPI_KEY`      | (empty)                         | SerpApi API key                                 |
| `OPENCHEF_FIRECRAWL_API_KEY`| (empty)                         | Firecrawl API key                               |

### Directory structure

The system creates the following directories automatically:

```
agent_memories/          # agent memory JSON files
backups/                 # database backups
datasets/                # dataset specification files and output
lineage/                 # (future) lineage tracking
logs/                    # openchef.log and error_log.md
long_messages/           # (future) long message storage
open_team/               # OpenTeamFormat support
research/                # Siphon research reports
skills/                  # skill scripts (Python, Bash, C, MAML)
stack_templates/         # stack template files
thread_repos/            # Git repositories for moderated code
workspace/               # temporary workspaces for skills
```

---

## Web Dashboard

The dashboard is a single‑page application that communicates with the backend via WebSocket.

### Tabs

-   **AGENTS** – view all agents, their status, model, and channels. Click an agent to edit its settings or delete it. Use the `SPAWN AGENT` button to create new agents.
-   **SKILLS** – browse all loaded skills, view their metadata, and use quick‑action buttons. The **RELOAD** button refreshes the skill registry; **CREATE** opens a dialog to generate a new skill via reverse engineering.
-   **RALPH** – monitor active Ralph sessions, view convergence and stagnation metrics, and start new brainstorming sessions.
-   **SIPHON** – start research sessions and view past research reports stored in IndexedDB.

### Chat Interface

-   Type any message in the input bar to send it to the current channel (`general`, `code`, `siphon`, or `ralph`).
-   All agent responses appear in the chat log.
-   Commands (starting with `/`) are processed by the server.

---

## CLI Mode

Run with `--cli` to disable the web UI. All commands are available through the terminal.

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/skills` | List loaded skills |
| `/agents` | List active agents |
| `/bio` | Show bio.md |
| `/soul` | Show soul.md |
| `/status` | Check Ollama connectivity |
| `/errorlog` | Show error log |
| `/siphon <query>` | Start a research session |
| `/dataset <name>` | Run a dataset extraction job |
| `/datasets` | List available dataset files |
| `/ralph <topic>` | Start an evolutionary brainstorming session |
| `/abstract <query>` | Generate a step‑by‑step plan |
| `/stack list` | List stack templates |
| `/stack build <name>` | Build a repository from a template |
| `/stack add <description>` | Create a template from description |
| `/stack import <json>` | Bulk import templates |
| `/spawn_interactive` | Interactive agent creation |
| `/update_agent <id> <field> <value>` | Update an agent |
| `/remove_agent <id>` | Remove an agent |
| `/reverse_skill <description>` | Generate a skill from description |
| `/quit` | Exit CLI |

---

## Core Concepts

### Agents

Agents are the primary actors in the system. Each agent has:

-   A unique ID
-   A name
-   A model (Ollama model name)
-   A system prompt
-   A list of channels it listens to
-   A status (`online`, `offline`, `working`)

Agents can be spawned, updated, and removed at runtime. The `moderator` agent is a special embedded agent that only operates on the `code` channel and performs linting and repository commits.

### Skills

Skills are reusable functions that agents can invoke. The SkillEngine supports:

-   **Python** – `# NAME:`, `# DESCRIPTION:`, and `# ARGS:` comments at the top.
-   **Bash** – same comment convention.
-   **C** – `// NAME:`, `// DESCRIPTION:`, `// ARGS:` comments.
-   **MAML** – Markdown files with YAML frontmatter defining name, version, description, triggers, capabilities, etc.

Skills are executed in an isolated temporary workspace with resource limits (memory, CPU time, output length). They can return JSON output for structured data.

### MAML & Reverse Skills

**MAML (Markdown + YAML)** skills are self‑describing. The frontmatter defines the skill metadata; the body contains the implementation (or documentation).

**Reverse Skill** – describe any task in plain English, and the system uses the LLM to generate a complete Python skill script. The generated script includes the required comments and outputs JSON.

### Siphon Research Engine

Siphon performs autonomous research:

1.  Accepts a query (or URL).
2.  Searches using DuckDuckGo (API and Lite HTML fallback) and optionally SerpApi.
3.  Scrapes the top URLs (using BeautifulSoup).
4.  Extracts facts from scraped content using an LLM.
5.  Generates a summary.
6.  Posts the full report to the `siphon` channel.

Research progress is broadcast via WebSocket, and reports are stored in the local filesystem and IndexedDB (browser).

### Ralph Evolutionary Engine

Ralph evolves project specifications over multiple generations:

-   **Initial population** – generate `population_size` specs.
-   **Evaluation** – each spec is scored by an LLM.
-   **Mutation** – the best spec is mutated to produce the next generation.
-   **Convergence detection** – stops when the best spec changes by less than `convergence_threshold`.
-   **Stagnation** – if convergence stalls for `stagnation_limit` generations, a forced mutation is injected.

Ralph sessions post progress reports to the `ralph` channel.

### Abstract Planner

The Abstract Engine generates structured plans for any request. Plans include:

-   Goal
-   Prerequisites
-   Numbered steps
-   Risks and mitigation
-   Expected outcome

Plans are posted as messages in the current channel.

### Dataset Engine

The Dataset Engine automates structured data extraction from web pages.

**Dataset Specification File** (Markdown with YAML frontmatter):

```markdown
---
dataset_id: example_dataset
target_embedding_model: nomic-embed-text
dimension: 768
chunk_size: 512
chunk_overlap: 64
storage_format: json
---

## Schema

| Field Name | Data Type | Description | Extraction Strategy / Selector |
|------------|-----------|-------------|--------------------------------|
| `title`    | string    | page title  | `h1.title`                     |
| `content`  | string    | main text   | `div.content`                  |

## Targets

- [ ] URL: `https://example.com/page1`
- [ ] URL: `https://example.com/page2`
```

The engine parses the schema, scrapes each target URL, and uses the LLM to extract the specified fields. Results are stored in JSON, JSONL, or ChromaDB.

### Stack Templates

The Stack Engine manages reusable project templates:

-   **List** – see all available templates.
-   **Build** – create a new repository from a template.
-   **Add** – describe a template in plain English and let the LLM generate it.
-   **Import** – bulk import templates from a JSON array.

### Moderator & Code Repositories

When an agent responds in the `code` channel, the Moderator:

1.  Extracts all code blocks (```language ... ```).
2.  Writes each block to a file in a Git repository (one repo per thread).
3.  Lints the code using appropriate tools:
    -   Python: `ruff` or `flake8`
    -   JavaScript: `biome` or `node -c`
    -   JSON: built‑in JSON parser
    -   HTML: `htmlhint`
    -   C/C++: `gcc -fsyntax-only`
4.  Commits the files to the repository (with or without errors).
5.  Posts feedback to the channel.

### DecentMem & J‑Space

-   **DecentMem** – each agent maintains its own memory pool:
    -   `e‑pool` – high‑score trajectories (≥60)
    -   `x‑pool` – low‑score trajectories (<60)
    -   Scores are assigned by an LLM judge.
    -   Retrieval uses cosine similarity on embeddings.
-   **J‑Space** – the system maintains a global workspace context: a list of key concepts extracted from recent messages. This context is injected into all agent prompts, promoting a shared understanding.

### Cron Engine

Cron jobs are stored in the database. Each job has a name, a cron schedule, and a webhook to call when triggered. If `croniter` is installed, the schedule is parsed; otherwise, jobs run every hour.

---

## API & WebSocket Commands

### WebSocket Commands (Client → Server)

| Command                     | Payload                                      | Description |
|-----------------------------|----------------------------------------------|-------------|
| `join`                      | `{ "channel": "general" }`                   | Join a channel |
| `set_username`              | `{ "username": "human" }`                    | Set display name |
| `message`                   | `{ "content": "Hello" }`                     | Send a message or command |
| `get_history`               | `{ "channel": "general" }`                   | Request message history |
| `spawn_agent`               | `{ "name", "model", "system_prompt", "channels" }` | Create a new agent |
| `update_agent`              | `{ "id", "name", "model", "system_prompt", "channels" }` | Update an agent |
| `remove_agent`              | `{ "id" }`                                   | Delete an agent |
| `siphon_start`              | `{ "query": "..." }`                         | Start a research session |
| `dataset_run`               | `{ "name": "..." }`                          | Run a dataset job |
| `dataset_list`              | `{}`                                         | List dataset files |
| `reload_skills`             | `{}`                                         | Reload the skill registry |
| `request_metrics`           | `{}`                                         | Force a metrics broadcast |
| `get_ralph_sessions`        | `{}`                                         | Get active Ralph sessions |
| `ralph_start`               | `{ "topic": "..." }`                         | Start a Ralph session |
| `abstract_start`            | `{ "query": "..." }`                         | Generate a plan |
| `get_siphon_history`        | `{}`                                         | Get past research reports |
| `close`                     | `{ "code": 1000, "reason": "..." }`          | Close the WebSocket |

### Server → Client Messages

| Type              | Payload                                      | Description |
|-------------------|----------------------------------------------|-------------|
| `new_message`     | `{ "channel", "message" }`                   | New chat message |
| `notification`    | `{ "msg": "..." }`                           | System notification |
| `error`           | `{ "msg": "..." }`                           | Error message |
| `skills_list`     | `{ "skills": [...] }`                        | List of loaded skills |
| `agents_list`     | `{ "agents": [...] }`                        | List of agents |
| `history`         | `{ "channel", "messages": [...] }`           | Message history |
| `metrics_update`  | `{ "agents": { ... } }`                      | Per‑agent metrics (CPU, memory, activity) |
| `global_metrics`  | `{ "cpu", "memory", "token_rate" }`          | Global system metrics |
| `agent_loading`   | `{ "agent_id", "loading": bool }`            | Agent activity indicator |
| `ralph_update`    | `{ "session_id", "data": {...} }`            | Ralph progress update |
| `ralph_sessions`  | `{ "sessions": [...] }`                      | Active Ralph sessions |
| `siphon_history`  | `{ "history": [...] }`                       | Research report history |
| `research_update` | `{ "session_id", "data": {...} }`            | Siphon progress update |

### REST Endpoints

| Endpoint         | Method | Description |
|------------------|--------|-------------|
| `/`              | GET    | Web UI |
| `/ws`            | GET    | WebSocket endpoint |
| `/api/metrics`   | GET    | Current agent metrics (JSON) |
| `/api/tags`      | GET    | List Ollama models (supports `?refresh=true`) |
| `/api/errorlog`  | GET    | Error log contents |
| `/health`        | GET    | Health check |

---

## Troubleshooting

### Common Issues

**Port 3721 already in use"**  
Stop any other instance of HUMBOLDT‑CHEF or change the port:
```bash
python openchef.py --port 3722
```

**"Ollama is not reachable"**  
- Ensure Ollama is running: `ollama serve`
- Check `OPENCHEF_OLLAMA_URL` in `.env` or environment.
- Try `curl http://localhost:11434/api/tags`.

**"Model X not found"**  
Pull the model:
```bash
ollama pull qwen2.5:0.5b
```

**"Skill execution timeout"**  
Increase `OPENCHEF_SKILL_TIMEOUT` in `.env`.

**"Database is locked"**  
The system uses WAL mode and busy timeout; if you see frequent locks, reduce concurrency or increase `OPENCHEF_DB_TIMEOUT` (hardcoded to 30s).

**"BeautifulSoup not installed"**  
Siphon scraping will be limited; install `beautifulsoup4`.

**"GitPython not installed"**  
Moderator commits are disabled; install `GitPython`.

**"chromadb initialization fails"**  
The dataset engine falls back to JSON storage; install `chromadb` for vector storage.

---

## License

MIT

Copyright (c) 2026 OPENCHEF!

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
