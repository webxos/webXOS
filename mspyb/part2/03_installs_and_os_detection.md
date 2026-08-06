# MSPYB v3.0 — Mandatory INSTALLS Section and OS Detection

Every MSPYB file begins its executable body (immediately after the docstring and imports) with an INSTALLS block. This block runs before any `write_file` calls. Its job is to detect the host operating system and basic device characteristics, then ensure the exact libraries required by the project are present at the correct versions.

In v3 the same detection results are also handed to the Auto-Start subsystem so that launch decisions can be platform-correct.

## Required Behaviour of INSTALLS

1. Detect OS family (Linux / macOS / Windows) and architecture (x86_64, arm64, …).
2. Detect presence of Python version, pip, and optionally Docker / Node / system package managers.
3. Declare a list of required packages with pinned versions (e.g. `fastapi==0.115.0`).
4. For each package: check if installed at the correct version; if not, install or upgrade via pip (or the appropriate system package manager when a binary dependency is required).
5. Log every install/upgrade decision through the Error Logging subsystem.
6. Abort with a clear message if a non-recoverable prerequisite is missing (e.g. no pip, no network when offline install is not possible).
7. Support an optional `--skip-installs` flag for air-gapped or CI environments that already provide the dependencies.
8. Expose the detection results (system, machine, python version, presence of Docker, etc.) in a structure that the later Auto-Start block can read.

## Platform Detection Helper

A minimal, reliable detector looks like this:

```python
def detect_environment() -> dict:
    """Return OS / architecture / Python / tool presence information."""
    system = platform.system()          # "Linux", "Darwin", "Windows"
    machine = platform.machine()        # "x86_64", "arm64", "AMD64", ...
    return {
        "system": system,
        "release": platform.release(),
        "machine": machine,
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "has_docker": shutil.which("docker") is not None,
        "has_compose": shutil.which("docker-compose") is not None or shutil.which("docker") is not None,
        "is_linux": system == "Linux",
        "is_macos": system == "Darwin",
        "is_windows": system == "Windows",
    }
```

The returned dictionary is stored in a module-level variable (commonly `ENV`) so that both the install logic and the later launch logic can consult it without re-running detection.

## Idempotent Package Installation

```python
REQUIRED_PACKAGES = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.6",
    "pydantic==2.8.2",
]

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
    env = detect_environment()
    log.info("Environment: %s", json.dumps(env))
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
        log.critical("Install failed: %s — recovery: check network, pip permissions, or use --skip-installs", exc)
        sys.exit(1)
```

## System-Level Dependencies

When a project needs non-Python binaries (for example a C compiler, a database client, or Docker), the INSTALLS block may also invoke the platform package manager:

- Linux: `apt-get`, `dnf`, or `pacman` (detected via `/etc/os-release`)
- macOS: Homebrew (`brew`)
- Windows: `winget` or Chocolatey when available

These steps remain optional and should be gated behind clear documentation in the header. The bootstrap must never assume elevated privileges without warning.

## Feeding Detection Results to Auto-Start

After `run_installs` finishes, the `ENV` dictionary is left in place. The Auto-Start block later reads keys such as `is_linux`, `is_macos`, `is_windows`, and `has_docker` to choose the correct spawn method. This tight coupling is intentional: the same information that decides how to install dependencies also decides how to start the resulting program.

## Flags

| Flag | Effect |
|------|--------|
| `--skip-installs` | Skip the package-installation phase only. Generation and Auto-Start still run. |
| `--no-launch` | Skip Auto-Start. Generation still occurs. |
| (both) | Pure generation mode — useful for CI that only needs the source tree. |

Both flags are parsed early, before any side effects, so that the rest of the script can branch cleanly.
