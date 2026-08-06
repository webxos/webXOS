# MSPYB v3.0 — Auto-Start / Auto-Launch Subsystem in Depth

The Auto-Start subsystem is the principal addition of version 3.0. Its purpose is to turn the bootstrap from a pure generator into a complete self-demonstrating cycle: install dependencies, materialize the project, and immediately run the resulting artifact on the host platform.

## Design Goals

- **Default-on.** Interactive users receive a running application without extra flags.
- **Explicit opt-out.** `--no-launch` (and the `AUTO_LAUNCH` template variable) give full control for automation.
- **Platform-correct.** Launch logic consults the same `ENV` dictionary produced by INSTALLS.
- **Observable.** Every decision and every failure is recorded with a recovery hint.
- **Non-blocking for long-running services.** The bootstrap may detach after a short grace period so that a web server or daemon continues running while the bootstrap process itself exits cleanly.
- **Idempotent.** Re-running the bootstrap does not leave orphaned processes unless the generated application itself is written that way.

## Decision Flow

```
parse flags (--no-launch, --skip-installs)
        │
        ▼
run INSTALLS (or skip)
        │
        ▼
generate project tree via write_file calls
        │
        ▼
AUTO_LAUNCH == "true"  AND  not --no-launch  ?
        │
       yes ──► determine launch method from ENV + TEMPLATE_VARS
        │         │
        │         ├─ Python module / script
        │         ├─ Docker Compose / Docker
        │         ├─ Native binary / shell script
        │         └─ Custom command
        │
        │         spawn process
        │         log outcome
        │
       no  ──► log "Auto-launch disabled"
        │
        ▼
print final status messages
```

## Launch Strategies by Project Type

### 1. Pure Python Application

```python
LAUNCH_COMMAND = "python -m app.main"
# or
LAUNCH_COMMAND = "python app/main.py"
```

The bootstrap changes into `PROJECT_ROOT` (or the subdirectory named by `LAUNCH_CWD`) and executes the command with the same Python interpreter that ran the bootstrap (`sys.executable`).

### 2. Docker Compose Stack

```python
LAUNCH_COMMAND = "docker compose up --build -d"
# or the older form
LAUNCH_COMMAND = "docker-compose up --build -d"
```

Before launching, the Auto-Start block should verify that Docker is available (`ENV["has_docker"]`). If Docker is missing it logs a critical recovery hint and aborts the launch attempt without failing the entire bootstrap.

### 3. Native Binary or Shell Script

When the generated project produces a compiled executable or a shell entry-point:

```python
LAUNCH_COMMAND = "./bin/myapp"          # Unix
LAUNCH_COMMAND = "bin\\myapp.exe"       # Windows (or let the platform logic adjust)
```

On Unix the bootstrap ensures the executable bit is set. On Windows it may need to adjust path separators.

### 4. Multi-step or Custom Launch

For more complex cases the `LAUNCH_COMMAND` can be a small shell one-liner, or the Auto-Start block can be extended with a project-specific function that the megalith author supplies. The canonical skeleton keeps the simple string form so that LLM agents can modify it without rewriting control flow.

## Platform-Specific Considerations

### Linux

- Prefer `subprocess.Popen` with the current environment.
- For GUI applications, ensure `DISPLAY` is present or document that a virtual framebuffer is required.
- Detach long-running services with a short `communicate(timeout=...)` then leave the child running.

### macOS

- Same process model as Linux.
- For `.app` bundles the `open` command may be more appropriate; the launch block can detect a `.app` suffix and switch strategy.
- Homebrew-installed tools are already on the PATH if INSTALLS used Homebrew.

### Windows

- Use `sys.executable` for Python entry points so that the correct interpreter is chosen.
- Path separators in `LAUNCH_COMMAND` should be written with forward slashes or constructed with `os.path.join` so they remain portable.
- Console allocation: for CLI tools it is usually desirable to inherit the current console; for GUI tools `CREATE_NEW_CONSOLE` or `DETACHED_PROCESS` flags can be applied.

## Error Handling and Recovery Hints

Typical failure modes and the hints the logger should emit:

| Failure | Recovery hint |
|---------|---------------|
| Command not found | Verify LAUNCH_COMMAND and that INSTALLS installed the required runtime |
| Port already in use | Change API_PORT in TEMPLATE_VARS or stop the conflicting process |
| Docker daemon not running | Start Docker Desktop / the system Docker service |
| Permission denied (executable) | Ensure the generated binary has the executable bit (Unix) |
| Missing shared library | Re-run with full INSTALLS or install the system package listed in the header |

All of these are recorded at ERROR or CRITICAL level together with the original exception so that both humans and LLM agents can diagnose the problem from the log alone.

## Graceful Degradation

If Auto-Start fails, the bootstrap still exits with a success status for the generation phase (unless a critical install error occurred earlier). The final print statements make it clear that the project was generated and that launch was attempted but did not succeed. This separation keeps the “generate” contract intact even when the “run” contract cannot be fulfilled on a particular machine.

## Configuration Surface

Everything that controls Auto-Start is visible in three places:

1. `TEMPLATE_VARS["AUTO_LAUNCH"]`, `["LAUNCH_COMMAND"]`, `["LAUNCH_CWD"]`
2. Command-line flags `--no-launch` and `--skip-installs`
3. The `ENV` dictionary produced by `detect_environment()`

An LLM agent that needs to change how a project starts has a single, well-documented set of knobs and does not need to invent new control-flow patterns.
