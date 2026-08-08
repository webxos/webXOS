# MSPYB — Skill

**Megalithic Singular Python Bootstrap**

Follow this skill to build:

One self-executing Python file defines, generates, and launches an entire multi-file project.


**Make sure to avoid these errors:**

``` error
I see you're still getting the same error – that means you're running the old version of the file that still uses triple double quotes (""") for the outer string in write_file.

The fix is simple: replace every """ that starts a multi‑line file content with ''' (triple single quotes). That way, inner """ (e.g., docstrings inside the generated code) won’t close the outer string prematurely.
```
As long as you build it in one concise py file it should work as intended to properly reboot the full system...

Here is the full guide


## 1. Philosophy

MSPYB is a pattern, not a framework. The single `bootstrap.py` is the living specification.

| Principle              | Meaning |
|------------------------|---------|
| Singular source of truth | One file holds the full project definition |
| Self-bootstrapping     | `python3 bootstrap.py` creates the tree after installing deps |
| LLM-native             | `LLM-INSTRUCTION` comments guide future agents |
| Megalithic             | Intentionally complete rather than fragmented |
| Executable documentation | The script is both the spec and the generator |
| Idempotent             | Re-running regenerates a clean project |
| Full-circle (v3)       | Install → Generate → Auto-Launch |

**v3 addition**: After generation the bootstrap automatically starts the application on the detected platform. The last gap between “generated” and “running” is closed.

Canonical invocation:

```bash
cd /path/to/folder
python3 bootstrap.py
```

Flags:

- `--skip-installs` — skip dependency installation
- `--no-launch` — generate only, do not start the app
- both together — pure generation mode

## 2. The Five Required Capabilities

A v3 megalith must implement all five.

### 2.1 Error Logging

- Bootstrap logger available **before** INSTALLS.
- Generate a shared `shared/logging_setup.py` used by every service.
- Prefer structured JSON lines + human console handler.
- Record every Auto-Start decision (attempted / succeeded / skipped / failed + recovery hint).

### 2.2 Templating

- Single `TEMPLATE_VARS` dictionary near the top.
- `render()` uses `string.Template.safe_substitute`.
- All `write_file` content may contain `$VAR` or `${VAR}`.
- Required Auto-Start keys: `AUTO_LAUNCH`, `LAUNCH_COMMAND`, `LAUNCH_CWD`.
- Never put secrets in `TEMPLATE_VARS`.

### 2.3 Versioning

- `MSPYB_VERSION = "X.Y.Z"` at the top.
- Generated `VERSION` file stays in sync.
- Header documents known fixes and the version in which each was resolved.
- Major version 3 signals the Auto-Start contract.

### 2.4 INSTALLS (OS / device aware)

Runs first, before any `write_file`.

1. Detect OS family, architecture, Python version, presence of Docker.
2. Expose results in module-level `ENV` dict (consumed later by Auto-Start).
3. Declare pinned packages in `REQUIRED_PACKAGES`.
4. Idempotent install via `pip` (or system package manager when needed).
5. Honour `--skip-installs`.
6. Log every decision; abort only on non-recoverable failure.

### 2.5 Auto-Start / Auto-Launch

Runs after generation.

1. Respect `AUTO_LAUNCH` and `--no-launch`.
2. Use `ENV` for platform decisions.
3. Change to `LAUNCH_CWD` (relative to `PROJECT_ROOT`).
4. Spawn via `subprocess.Popen`; short grace period for long-running services.
5. On failure log a recovery hint; **do not** treat launch failure as generation failure.
6. Final print statements make clear whether launch was attempted.

## 3. Canonical Structure

See the skeleton in `SKILL.md`. Logical order of the file:

1. Shebang + rich docstring (version, LLM-INSTRUCTION, architecture, known fixes, usage)
2. Imports
3. Versioning constants
4. `TEMPLATE_VARS`
5. Bootstrap logger
6. INSTALLS block (`REQUIRED_PACKAGES`, `detect_environment`, `run_installs`)
7. Flag parsing + `run_installs(...)`
8. `render` + `write_file` helpers
9. Generation sections (root → shared → app → services)
10. `auto_launch()` function + call
11. Final status prints

## 4. Auto-Start Decision Flow

```
parse flags
    │
    ▼
run INSTALLS (or skip)
    │
    ▼
generate project tree
    │
    ▼
AUTO_LAUNCH == "true"  AND  not --no-launch  ?
    │
   yes ──► determine command from TEMPLATE_VARS + ENV
    │         spawn process
    │         log outcome
    │
   no  ──► log "Auto-launch disabled"
    │
    ▼
print final status
```

### Launch strategies

| Project type       | Typical LAUNCH_COMMAND                                      |
|--------------------|-------------------------------------------------------------|
| Pure Python        | `python -m app.main`                                        |
| FastAPI / ASGI     | `uvicorn app.main:app --host 0.0.0.0 --port $API_PORT`     |
| Docker Compose     | `docker compose up --build -d`                              |
| CLI tool           | `python -m app.cli --help`                                  |
| Multi-service      | custom function or ordered shell one-liner                  |

Platform notes:

- Prefer `sys.executable` for Python entry points.
- On Unix ensure executable bit for native binaries.
- Detach long-running services after a short `communicate(timeout=...)`.
- Guard Docker launches with `ENV["has_docker"]`.

### Common recovery hints

| Failure                  | Hint |
|--------------------------|------|
| Command not found        | Verify LAUNCH_COMMAND and that INSTALLS installed the runtime |
| Port already in use      | Change API_PORT or stop the conflicting process |
| Docker daemon not running| Start Docker Desktop / system Docker service |
| Permission denied        | Ensure executable bit (Unix) |
| Missing shared library   | Re-run full INSTALLS or install the system package |

## 5. Extension Points

- **Custom launch function** — replace the string command with a callable that receives `ENV`.
- **Health-check gate** — poll `/health` before declaring success.
- **Platform-specific commands** — keys such as `LAUNCH_COMMAND_LINUX` selected by `ENV`.
- **Post-launch hooks** — open browser, print banner, etc.
- **Multi-process orchestration** — launch API + worker + frontend in order; generate a matching `stop.sh`.

## 6. Workflow & Best Practices

1. Author / evolve the single megalith.
2. Run full-circle: `python3 bootstrap.py`.
3. Configure secrets once: `cp project/.env.example project/.env`.
4. Iterate by editing only the bootstrap; re-run.
5. Document every significant fix in the header and bump `MSPYB_VERSION`.

Checklist for a production-ready megalith:

- [ ] Rich header with version, architecture, LLM-INSTRUCTION, usage (both flags), known fixes
- [ ] `MSPYB_VERSION` + generated `VERSION`
- [ ] `TEMPLATE_VARS` including the three Auto-Start keys
- [ ] Bootstrap logger before any side effects
- [ ] INSTALLS with OS detection, pinned packages, `--skip-installs`, `ENV`
- [ ] `write_file` + templating
- [ ] Shared logging module
- [ ] Auto-Start block that respects flags and logs outcomes
- [ ] Clear final print statements
- [ ] No secrets in the megalith

## 7. Why This Works Extremely Well with LLMs

- Single context window contains the whole system definition.
- `LLM-INSTRUCTION` comments act as durable system prompts.
- Changing one value in `TEMPLATE_VARS` updates every generated file and the launch behaviour.
- Launch failures are logged with recovery hints; an agent can diagnose, edit the megalith, and re-run without manual steps.
- The agent’s job shrinks to “improve the megalith”; the bootstrap itself proves the improvement works.

## 8. Concrete Examples

### Minimal FastAPI service

```python
REQUIRED_PACKAGES = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.6",
]
TEMPLATE_VARS = {
    ...
    "LAUNCH_COMMAND": "uvicorn app.main:app --host 0.0.0.0 --port $API_PORT",
}
# generate app/main.py that defines FastAPI() with a /health route
```

### Docker Compose stack

```python
TEMPLATE_VARS["LAUNCH_COMMAND"] = "docker compose up --build -d"
# also write a docker-compose.yml that uses the same ports / tags
```

### Headless / library project

```python
TEMPLATE_VARS["AUTO_LAUNCH"] = "false"
```

## Final Remark

Treat the bootstrap script as the living specification of your system. Everything else is generated from it — and, in v3, started by it.

Once you have studied this idea we will proceed to build this full python file in MSPYB format...
