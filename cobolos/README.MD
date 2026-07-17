# COBOLOS – Local COBOL Modernization Agent Harness (Under Development v2.0) 

```
   ____      _           _  ___  ____  
  / ___|___ | |__   ___ | |/ _ \/ ___| 
 | |   / _ \| '_ \ / _ \| | | | \___ \ 
 | |__| (_) | |_) | (_) | | |_| |___) |
  \____\___/|_.__/ \___/|_|\___/|____/ 
  
```

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-required-orange.svg)](https://ollama.ai/)

COBOLOS is a self‑contained, local AI‑powered assistant for mainframe modernization. It provides a set of ten specialized agents that leverage large language models to help you understand, transform, and document COBOL code, JCL, and mainframe artifacts.

---

## Features

- **10 Specialized Agents** – From natural‑language SQL generation to impact analysis and automated documentation.
- **Web UI & Terminal TUI** – Choose your preferred interface; both offer full agent functionality.
- **Streaming Responses** – See agent outputs token‑by‑token via Server‑Sent Events (SSE) or WebSockets.
- **Optional API Key Authentication** – Secure your endpoints with a simple API key.
- **External Database Support** – Connect to DB2, VSAM (fallback to SQLite) for Agent 01 (NL Data Query).
- **Cron‑Scheduled Agent Runs** – Automate recurring tasks with built‑in scheduler.
- **Local Workspace & History** – All artifacts and interaction history stored locally; optional Git versioning.
- **File Upload & Export** – Upload COBOL source files and export your entire workspace as a ZIP.
- **Automatic Bootstrap** – Creates a virtual environment and installs dependencies on first run, even on externally‑managed Python (Kali, Ubuntu 24.04+).

---

## Requirements

- **Python 3.8 or higher** (3.10+ recommended)
- **Ollama** installed and running locally with at least one model

---

## Installation

COBOLOS is **self‑bootstrapping** single Python file. Simply download the file and place it in a folder on your system:

Then run:

```bash
cd ~/cobolos/
python3 cobolos.py
```

On first run, the script will:
- Detect missing Python packages.
- If your system uses an externally‑managed Python environment (e.g., Kali, Ubuntu 24.04+), it will automatically create a `venv` directory, install all dependencies inside it, and re‑execute itself using the virtual environment's Python.
- If packages are missing but the environment is not externally managed, it installs them globally (or in the current venv) and continues.

**All dependencies are installed automatically** – no manual `pip install` required.

---

## Usage

### 1. Web UI Dashboard

Run the script without arguments to launch the web interface:

```bash
python3 cobolos.py
```

- Opens `http://127.0.0.1:8080` in your default browser.
- Select an agent, enter your input, and choose **Run**.
- Toggle **Stream** to see live token‑by‑token responses.
- Upload COBOL/JCL files, browse workspace, view history, and export your workspace.

### 2. Terminal TUI Command Examples

For a rich terminal interface:

```bash
python3 cobolos.py --tui
```

You'll get an interactive agent selection menu with formatted output.

### 3. Command‑Line (CLI) Mode

Run a single agent headlessly:

```bash
python3 cobolos.py --cli --agent 01 --input "Show all customers from California"
```

Optionally save the JSON result to a file:

```bash
python3 cobolos.py --cli --agent 04 --input "$(cat myprogram.cob)" --output result.json
```

### 4. Server Options

Change host/port:

```bash
python3 cobolos.py --host 0.0.0.0 --port 9000
```

---

## Configuration

The configuration file `config.json` is created in the same directory as `cobolos.py` on first run. You can edit it to customise:

- `ollama_url`: Ollama API endpoint (default `http://localhost:11434/api/generate`)
- `default_model`: Default model name (e.g., `codellama:7b`, `llama3`)
- `temperature`, `max_tokens`: Generation parameters
- `timeout`, `agent_timeout`, `stream_timeout`: Timeouts in seconds
- `api_key`: Set to a string to enable authentication (clients must send `X-API-Key` header or `api_key` query param)
- `external_db`: Configure connection to external DB (type: `sqlite`, `db2`, or `vsam`; connection string as needed)
- `schedules`: Array of cron jobs, e.g.:
  ```json
  "schedules": [
    {"agent_id": "02", "cron": "0 9 * * *", "input": "Check system logs for abends"}
  ]
  ```

You can also update the model and other settings via the Web UI's **Controls** panel.

---

## Agents Overview

| ID | Name | Description |
|----|------|-------------|
| 01 | NL Data Query | Convert natural language questions into safe SQL queries (supports external DB) |
| 02 | Error Log Diagnostic | Analyse mainframe abend logs, suggest fixes, reference IBM error codes |
| 03 | JCL/Batch Generator | Generate JCL job streams from a high‑level description |
| 04 | Code Explainer | Explain COBOL code: purpose, paragraph breakdown, data flow, Mermaid flowchart |
| 05 | Automated Unit Test | Generate Python `pytest` harness for COBOL programs |
| 06 | Modernization Guard | Recommend refactoring options (COBOL OO, Java, C#, Python) |
| 07 | 3270 Screen API | Turn a 3270 screen layout into an OpenAPI specification and Python client |
| 08 | Rule Extraction | Extract business rules as decision tables and functional groups |
| 09 | Smart Impact Analysis | Search workspace for dependencies and generate risk scores |
| 10 | Auto Documentation | Create comprehensive documentation in Markdown, HTML, and PDF (base64) |

---

## API Endpoints

The web UI is built on a FastAPI backend. You can interact programmatically:

- `GET /` – Web UI (HTML)
- `POST /api/agent/{agent_id}` – Run agent (JSON body: `{"input": "..."}`)
- `GET /api/agent/stream/{agent_id}?input=...` – Stream agent output via SSE
- `POST /api/upload` – Upload a file (multipart/form-data)
- `GET /api/workspace` – List workspace files
- `GET /api/history` – Retrieve interaction history (limit parameter)
- `GET /api/config` – Get current configuration
- `POST /api/config` – Update configuration
- `GET /api/export` – Download workspace as ZIP
- `WebSocket /ws/agent/{agent_id}` – Bi‑directional streaming with authentication support

**Authentication:** If `api_key` is set in `config.json`, all HTTP endpoints (except WebSocket handshake) require the `X-API-Key` header. WebSocket clients must pass `?api_key=...` in the URL or include `X-API-Key` in the WebSocket headers (if supported).

---

## Workspace & Data

All data is stored locally under `cobolos_workspace/`:

- `source_pool/` – Uploaded source files
- `artifacts/` – JSON results from agent runs
- `history/` – Optional raw history (currently SQLite)
- `cobolos.db` – Mock database for Agent 01 (customers/orders)
- `history.db` – Interaction history

If Git is installed, the workspace is automatically version‑controlled (commits on uploads and agent results).

---

## Troubleshooting

### “Error: Could not connect to Ollama”
- Ensure Ollama is running: `ollama serve`
- Verify `ollama_url` in `config.json` points to the correct endpoint.
- Pull a model: `ollama pull codellama:7b`

### “Form data requires python-multipart”
- The bootstrap should have installed `python-multipart`. If not, install it manually: `pip install python-multipart` (or recreate the venv).

### Web UI fails with `KeyError: ' background'`
- This is fixed in v2.0. If you see this, update to the latest version.

### Permission errors on Linux/macOS
- If you run into permission issues with `pip` or `venv`, try running with `--user` or as a non‑root user. The bootstrap will automatically use a venv to avoid system‑wide conflicts.

### Agent timeouts
- Increase `agent_timeout` in `config.json` if your model is slow.

---

## 📜 License

This project is licensed under the MIT License
