# MSPYB v3.0 — Extension Points and Concrete Examples

## Extension Points

The canonical skeleton is intentionally minimal so that project-specific needs can be added without fighting the structure.

### 1. Custom Launch Functions

Replace the simple string-based `LAUNCH_COMMAND` with a callable:

```python
def project_specific_launch(env: dict) -> None:
    if env["has_docker"]:
        subprocess.check_call(["docker", "compose", "up", "--build", "-d"], cwd=PROJECT_ROOT)
    else:
        subprocess.Popen([sys.executable, "-m", "app.main"], cwd=PROJECT_ROOT)

# then call project_specific_launch(ENV) inside auto_launch()
```

### 2. Multi-Service Orchestration

When the generated project contains several processes (API, worker, frontend), the Auto-Start block can launch them in dependency order and record each PID. A small `stop.sh` can later be generated to tear the set down cleanly.

### 3. Health-Check Gate

Before declaring launch successful, the bootstrap can poll a health endpoint:

```python
import time, urllib.request
for _ in range(10):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{TEMPLATE_VARS['API_PORT']}/health")
        log.info("Health check passed")
        break
    except Exception:
        time.sleep(0.5)
else:
    log.warning("Health check did not pass within timeout")
```

### 4. Platform Overrides Inside TEMPLATE_VARS

```python
TEMPLATE_VARS = {
    ...
    "LAUNCH_COMMAND_LINUX": "python -m app.main",
    "LAUNCH_COMMAND_MACOS": "python -m app.main",
    "LAUNCH_COMMAND_WINDOWS": "python -m app.main",
}
```

The Auto-Start block then selects the appropriate key based on `ENV`.

### 5. Post-Launch Hooks

A list of callables can be registered and executed after a successful spawn (for example, opening a browser to the local URL on desktop platforms).

## Concrete Example — Minimal Web Service

A complete, ready-to-run megalith for a FastAPI service would set:

```python
REQUIRED_PACKAGES = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.6",
]

TEMPLATE_VARS = {
    "PROJECT_NAME": "helloapi",
    "API_PORT": "8000",
    "LOG_LEVEL": "INFO",
    "MSPYB_VERSION": MSPYB_VERSION,
    "AUTO_LAUNCH": "true",
    "LAUNCH_COMMAND": "uvicorn app.main:app --host 0.0.0.0 --port $API_PORT",
    "LAUNCH_CWD": ".",
}
```

and generate an `app/main.py` that defines a FastAPI application with a `/health` route. Running `python3 bootstrap.py` then produces a listening HTTP server on the chosen port.

## Concrete Example — Docker Compose Stack

```python
TEMPLATE_VARS = {
    ...
    "AUTO_LAUNCH": "true",
    "LAUNCH_COMMAND": "docker compose up --build -d",
    "LAUNCH_CWD": ".",
}
```

The INSTALLS block can optionally verify that the Docker CLI is present and log a clear recovery hint if it is missing. Generation writes a `docker-compose.yml` that references the same port and image tags stored in `TEMPLATE_VARS`.

## Concrete Example — CLI Tool

```python
TEMPLATE_VARS = {
    ...
    "AUTO_LAUNCH": "true",
    "LAUNCH_COMMAND": "python -m app.cli --help",
    "LAUNCH_CWD": ".",
}
```

Auto-Start runs the help command (or a smoke-test sub-command) so the developer immediately sees that the entry point is functional.

## When to Disable Auto-Start by Default

Some projects are deliberately headless or are intended only as libraries. In those cases set:

```python
"AUTO_LAUNCH": "false",
```

in `TEMPLATE_VARS`. The `--no-launch` flag remains available for one-off overrides, and the rest of the v3 contract (logging, templating, versioning, INSTALLS) continues to apply.

## Final Remarks

MSPYB v3.0 completes the original vision of a singular, self-executing project definition. By making Auto-Start mandatory, the format guarantees that every compliant bootstrap not only describes and materializes a system but also demonstrates that the system works on the machine where it is invoked.

The single file remains the contract. Everything else — directory tree, configuration, source, documentation, and now the running process itself — is derived from it.

Treat the bootstrap script as the living specification of your system; everything else is generated from it and started by it.
