# MSPYB v3.0 — Canonical Structure of a Bootstrap File

The skeleton below is the minimal complete form for a v3 megalith. All four required enhancements (Error Logging, Templating, Versioning, Auto-Start) and the INSTALLS section are present.

```python
#!/usr/bin/env python3
"""
MSPYB -- Megalithic Singular Python Bootstrap
Version: 3.0.0

LLM-INSTRUCTION: High-level architecture overview, known fixes,
usage instructions, generated file tree, and guidance for future
LLM agents. Always keep Error Logging, Templating, Versioning,
and Auto-Start in sync when extending the system.

Architecture Overview:
- Shared logging + templating + versioning modules
- INSTALLS block detects OS and pins dependencies
- Auto-Start block launches the generated application on the detected platform
- ...

Known fixes (this version):
- ...

Usage:
    python3 bootstrap.py                 # full install + generate + auto-launch
    python3 bootstrap.py --skip-installs # skip deps, still generate + launch
    python3 bootstrap.py --no-launch     # install + generate, do not launch
    python3 bootstrap.py --skip-installs --no-launch
"""

from __future__ import annotations
import os
import sys
import platform
import subprocess
import shutil
import json
import logging
from datetime import datetime, timezone
from string import Template
from pathlib import Path
from typing import Dict, List, Optional

# ------------------------------------------------------------------
# Versioning (required)
# ------------------------------------------------------------------
MSPYB_VERSION = "3.0.0"
PROJECT_ROOT = "project"

# ------------------------------------------------------------------
# Templating vars (required) -- single source for all substitutions
# ------------------------------------------------------------------
TEMPLATE_VARS: Dict[str, str] = {
    "PROJECT_NAME": "myapp",
    "API_PORT": "8000",
    "LOG_LEVEL": "INFO",
    "MSPYB_VERSION": MSPYB_VERSION,
    "AUTO_LAUNCH": "true",          # "true" / "false"
    "LAUNCH_COMMAND": "python -m app.main",  # or "docker compose up", etc.
    "LAUNCH_CWD": ".",              # relative to PROJECT_ROOT
}

# ------------------------------------------------------------------
# Error Logging bootstrap (required) -- available before INSTALLS
# ------------------------------------------------------------------
def _setup_bootstrap_logger() -> logging.Logger:
    logger = logging.getLogger("mspyb")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = _setup_bootstrap_logger()

# ------------------------------------------------------------------
# INSTALLS (required) -- runs first, OS/device aware
# ------------------------------------------------------------------
REQUIRED_PACKAGES = [
    # "fastapi==0.115.0",
    # "uvicorn[standard]==0.30.6",
]

def detect_environment() -> dict:
    system = platform.system()
    return {
        "system": system,
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "has_docker": shutil.which("docker") is not None,
        "is_linux": system == "Linux",
        "is_macos": system == "Darwin",
        "is_windows": system == "Windows",
    }

ENV = detect_environment()

def package_installed(spec: str) -> bool:
    name = spec.split("==")[0].split("[")[0]
    try:
        import importlib.metadata as meta
        installed = meta.version(name)
        if "==" in spec:
            return installed == spec.split("==")[1]
        return True
    except Exception:
        return False

def run_installs(skip: bool = False) -> None:
    log.info("Environment: %s", json.dumps(ENV))
    if skip:
        log.info("Skipping installs (--skip-installs)")
        return
    if not REQUIRED_PACKAGES:
        log.info("No packages declared in REQUIRED_PACKAGES")
        return
    missing = [p for p in REQUIRED_PACKAGES if not package_installed(p)]
    if not missing:
        log.info("All required packages already satisfied")
        return
    log.info("Installing: %s", ", ".join(missing))
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + missing
    try:
        subprocess.check_call(cmd)
        log.info("Installs completed successfully")
    except subprocess.CalledProcessError as exc:
        log.critical("Install failed: %s", exc)
        sys.exit(1)

# Parse flags early
SKIP_INSTALLS = "--skip-installs" in sys.argv
NO_LAUNCH = "--no-launch" in sys.argv
run_installs(skip=SKIP_INSTALLS)

# ------------------------------------------------------------------
# Core helpers: write_file + render (Templating)
# ------------------------------------------------------------------
def render(content: str, extra: Optional[Dict[str, str]] = None) -> str:
    vars_ = {**TEMPLATE_VARS, **(extra or {})}
    try:
        return Template(content).safe_substitute(vars_)
    except Exception as exc:
        log.error("Template render failed: %s", exc)
        raise

def write_file(path: str, content: str, extra_vars: Optional[Dict[str, str]] = None):
    rendered = render(content, extra_vars)
    full_path = os.path.join(PROJECT_ROOT, path)
    parent = os.path.dirname(full_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(rendered)
    log.debug("Wrote %s", full_path)

# ------------------------------------------------------------------
# 1. Root files
# ------------------------------------------------------------------
write_file("VERSION", """$MSPYB_VERSION\n""")
write_file(".env.example", """LOG_LEVEL=$LOG_LEVEL\nAPI_PORT=$API_PORT\n""")
write_file("README.md", """# $PROJECT_NAME\n\nGenerated by MSPYB $MSPYB_VERSION\n""")

# ------------------------------------------------------------------
# 2. Shared utilities (Error Logging)
# ------------------------------------------------------------------
write_file("shared/__init__.py", "")
write_file("shared/logging_setup.py", """
# LLM-INSTRUCTION: Extend this logger for new services.
# Always import get_logger from here; do not create ad-hoc loggers.
import logging, json, sys
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_DIR / "app.jsonl")
        fh.setFormatter(JsonFormatter())
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)
    return logger
""")

# ------------------------------------------------------------------
# 3. Application / services ...
# ------------------------------------------------------------------
write_file("app/main.py", """
# LLM-INSTRUCTION: Entry point. Keep sys.path / env wiring consistent.
from shared.logging_setup import get_logger
log = get_logger("app")
log.info("Starting $PROJECT_NAME v$MSPYB_VERSION")
print("Hello from $PROJECT_NAME")
""")

# ------------------------------------------------------------------
# 4. Auto-Start / Auto-Launch (required in v3)
# ------------------------------------------------------------------
def auto_launch() -> None:
    if NO_LAUNCH or TEMPLATE_VARS.get("AUTO_LAUNCH", "true").lower() != "true":
        log.info("Auto-launch disabled")
        return

    cwd = os.path.join(PROJECT_ROOT, TEMPLATE_VARS.get("LAUNCH_CWD", "."))
    cmd_str = TEMPLATE_VARS.get("LAUNCH_COMMAND", "python -m app.main")
    cmd = cmd_str.split()

    log.info("Auto-launching: %s (cwd=%s)", cmd_str, cwd)
    try:
        # Platform-aware adjustments can be inserted here
        if ENV["is_windows"]:
            # Windows-specific handling if needed
            pass
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Optional: wait briefly and surface early output
        try:
            out, _ = proc.communicate(timeout=3)
            if out:
                print(out)
        except subprocess.TimeoutExpired:
            log.info("Process is running (detached after grace period)")
            # Leave the process running
    except Exception as exc:
        log.error("Auto-launch failed: %s — recovery: check LAUNCH_COMMAND and dependencies", exc)

auto_launch()

print(f"MSPYB {MSPYB_VERSION} bootstrap complete -- project generated in '{PROJECT_ROOT}'")
if not NO_LAUNCH:
    print("Application launch attempted (see log for details)")
else:
    print("Next steps: cd project && ...")
```

## Required Elements Checklist (v3.0)

- Shebang + rich module docstring containing version, LLM-INSTRUCTION, architecture, known fixes, and exact usage including `--skip-installs` and `--no-launch`.
- `MSPYB_VERSION` constant and generated `VERSION` file.
- `TEMPLATE_VARS` dictionary that now includes `AUTO_LAUNCH`, `LAUNCH_COMMAND`, and `LAUNCH_CWD`.
- Bootstrap logger available before INSTALLS.
- INSTALLS section with OS/device detection, pinned packages, idempotent install, and exposure of the `ENV` dictionary.
- `write_file` helper that applies templating then writes UTF-8 content.
- Logical sections grouped by responsibility.
- Shared logging module generated into the project.
- **Auto-Start block that runs after generation, respects flags, and uses platform information.**
- Final print statements confirming success and indicating whether launch was attempted.
