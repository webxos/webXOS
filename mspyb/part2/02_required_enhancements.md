# MSPYB v3.0 — The Four Required Enhancements

Every MSPYB agent and every project it generates must embed four capabilities. They are not optional add-ons; they form the canonical structure of a v3 megalith.

## 1. Error Logging (carried forward from v2)

A shared logging module is always generated. It writes structured JSON (or plain text) logs with timestamp, severity, module, message, optional exception traceback, and a short recovery hint. Logs land in `logs/` (or a configurable path). Critical failures also print a human-readable summary to stderr so the bootstrap itself surfaces problems immediately.

- Severity levels: DEBUG · INFO · WARNING · ERROR · CRITICAL
- Every generated service imports the shared logger.
- Bootstrap-time failures are logged before the process exits.
- `LLM-INSTRUCTION` comments in the logger explain how future agents should extend it.
- The bootstrap logger itself must be available before the INSTALLS block runs.

In v3 the logging subsystem additionally records every Auto-Start decision (launch attempted, launch succeeded, launch skipped, launch failed with exit code, etc.).

## 2. Templating (carried forward from v2)

A lightweight templating layer (stdlib `string.Template`) is embedded so that generated files can contain placeholders that are expanded at bootstrap time from a single `TEMPLATE_VARS` dictionary. This keeps environment-specific values (ports, image tags, service names, paths, launch commands) out of duplicated strings and makes the megalith itself the single place to change them.

- All `write_file` content may use `$VAR` or `${VAR}` placeholders.
- `TEMPLATE_VARS` is defined near the top of the megalith (after versioning, before INSTALLS).
- A `render()` helper substitutes variables before writing.
- Missing variables surface clearly; required keys can be validated explicitly.
- In v3, `TEMPLATE_VARS` also holds launch-related keys such as `LAUNCH_COMMAND`, `LAUNCH_CWD`, and `AUTO_LAUNCH`.

## 3. Versioning (carried forward from v2)

The megalith carries an explicit semantic version. That version is written into a generated `VERSION` file, into package metadata where applicable, and into the module docstring. Dependency versions are pinned inside the INSTALLS section so that re-bootstrapping always yields the same library set. A short known-fixes list lives in the header so agents can see the evolution of the artifact.

- `MSPYB_VERSION = "X.Y.Z"` at the top of the script.
- Generated `VERSION` file and `__version__` attributes stay in sync.
- INSTALLS pins exact package versions (or compatible ranges).
- Header documents known fixes and the version in which each was resolved.
- v3 bumps the major version to signal the addition of the mandatory Auto-Start contract.

## 4. Auto-Start / Auto-Launch (new in v3)

After the project tree has been successfully written, the bootstrap automatically starts the generated application using platform-aware logic. This is the defining addition of version 3.0.

### Required behaviour

1. Detect the host platform (Linux, macOS/Darwin, Windows) and architecture.
2. Determine the appropriate launch method for the kind of project that was generated (Python module, shell script, Docker Compose stack, native binary, etc.).
3. Honour the `AUTO_LAUNCH` value from `TEMPLATE_VARS` and the `--no-launch` command-line flag.
4. Change into the correct working directory if required.
5. Spawn the process, capture its initial output if desired, and log the outcome.
6. On failure, record a recovery hint (missing binary, port already in use, Docker daemon not running, etc.).
7. Never block indefinitely; for long-running services the bootstrap may either detach or wait a short grace period and then exit while leaving the child running.

### Default policy

- Auto-launch is **enabled by default**.
- `--no-launch` disables it for the current run.
- `--skip-installs` continues to skip only the dependency-installation phase; generation and launch still occur unless `--no-launch` is also supplied.

### Why it is mandatory

Without Auto-Start the developer (or the LLM agent) still has to remember a second command after the bootstrap finishes. That second command is frequently platform-specific and error-prone. By making launch part of the megalith itself, the single-file artifact becomes a complete, self-demonstrating system. The same script that defines the project also proves that the project works.

## Interaction Among the Four Enhancements

- INSTALLS supplies the platform information that Auto-Start needs.
- Templating supplies the concrete launch command and working directory.
- Error Logging records every step of the launch attempt.
- Versioning records which revision of the Auto-Start logic produced a given binary or service.

A megalith that implements only three of the four enhancements is not v3-compliant. All four must be present and functional.
