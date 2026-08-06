# MSPYB v3.0 — Recommended Workflow and Best Practices

## Recommended Workflow

1. **Author / evolve the megalith**  
   Edit `bootstrap.py` (or ask an LLM). Keep the header, `LLM-INSTRUCTION`, `TEMPLATE_VARS` (including the new Auto-Start keys), `REQUIRED_PACKAGES`, and known-fixes accurate.

2. **Bootstrap with full circle**  
   ```bash
   cd /path/to/folder
   python3 bootstrap.py
   ```
   This performs installs, generation, and auto-launch in one step.

3. **Alternative invocations**  
   ```bash
   python3 bootstrap.py --skip-installs          # deps already present
   python3 bootstrap.py --no-launch              # generate only
   python3 bootstrap.py --skip-installs --no-launch
   ```

4. **Configure secrets**  
   After the first successful generation:
   ```bash
   cd project
   cp .env.example .env
   # edit secrets
   ```
   Subsequent re-runs of the bootstrap preserve `.env` if the author has written the generation logic to avoid overwriting it.

5. **Iterate**  
   - Discover bugs or missing features.
   - Update the corresponding `write_file(...)`, `TEMPLATE_VARS`, `REQUIRED_PACKAGES`, or launch command.
   - Re-run `python3 bootstrap.py`.
   - Document the fix in the header and bump `MSPYB_VERSION`.

6. **Extend with an LLM**  
   Example prompt:
   > “Add a new billing service following the existing MSPYB v3 pattern. Update the header, TEMPLATE_VARS (including any new launch needs), INSTALLS, docker-compose, create the new service files with logging, and ensure Auto-Start still launches the whole stack.”

## Best Practices and Conventions

### Header

- Always include version, architecture overview, known fixes, exact usage (both flags), and the generated file tree.
- Keep the `LLM-INSTRUCTION` block up to date.
- Document every significant fix with Symptom / Location / Root cause / Exact fix / Impact / Version.

### INSTALLS

- Pin versions explicitly (`package==X.Y.Z`).
- Prefer pure-Python packages; document system-level prerequisites in the header and README.
- Detect at least: OS family, architecture, Python version, presence of Docker.
- Log every install decision at INFO; failures at CRITICAL and exit non-zero.
- Honour `--skip-installs` for CI and offline use.
- Store the detection result in a module-level `ENV` for Auto-Start to consume.

### Error Logging

- Generate a shared logger used by every service.
- Prefer structured JSON lines + a human console handler.
- Bootstrap-time logger must work even before the project `logs/` directory exists.
- Record Auto-Start decisions with the same severity conventions.

### Templating

- Keep `TEMPLATE_VARS` small and focused on values that truly vary.
- Never put secrets in `TEMPLATE_VARS`; secrets belong in `.env` only.
- Use `safe_substitute` so missing keys surface as visible `$PLACEHOLDERS`.
- Include the three Auto-Start keys: `AUTO_LAUNCH`, `LAUNCH_COMMAND`, `LAUNCH_CWD`.

### Versioning

- Bump `MSPYB_VERSION` on every meaningful change.
- Write the same version into the generated `VERSION` file and any `__version__` attributes.
- Keep a short known-fixes / changelog section in the docstring.
- Major version 3 signals the presence of the Auto-Start contract.

### Auto-Start

- Default to enabled.
- Provide a clear `--no-launch` escape hatch.
- Choose a `LAUNCH_COMMAND` that works on all three major platforms or document platform-specific overrides.
- Log the exact command and working directory that will be used.
- Prefer a short grace period for long-running services so the bootstrap can exit while leaving the child alive.
- Never treat a launch failure as a generation failure; keep the two phases independent.

### Naming

- Script: `bootstrap.py` (or `mspyb.py`, `megalith.py`)
- Output directory: `project/` (or configurable via a constant)
- Use clear section separators with `# ------------------------------------------------------------------`

### Security and Production Notes

- Never hard-code real secrets; always use `.env.example`.
- Document CORS, rate limits, JWT secrets, etc., as configuration that must be tightened for production.
- When Auto-Start launches a network service, remind the user (in the final print statements or the README) that the service is bound to the configured port and may be reachable on the local network.

## Summary Checklist for a Production-Ready MSPYB v3

- [ ] Rich header with version + architecture + LLM instructions + usage (both flags) + known fixes
- [ ] `MSPYB_VERSION` constant and generated `VERSION` file (Versioning)
- [ ] `TEMPLATE_VARS` + `render()` helper, including Auto-Start keys (Templating)
- [ ] Bootstrap logger available before any side effects (Error Logging)
- [ ] INSTALLS section: OS/device detection, pinned packages, idempotent install, `--skip-installs`, `ENV` dictionary
- [ ] `write_file` helper that templates then writes UTF-8 content
- [ ] Logical section comments (root → shared → services → Auto-Start)
- [ ] `.env.example`, requirements or `docker-compose.yml`, `README.md`
- [ ] Shared logging module with structured output and `LLM-INSTRUCTION`
- [ ] **Auto-Start block that respects flags, uses platform detection, and logs outcomes**
- [ ] Consistent path handling
- [ ] Working healthchecks or clean CLI startup banner
- [ ] All previously discovered bugs fixed and documented in the header
- [ ] Clear final print statements indicating whether launch was attempted
- [ ] Version bumped when significant fixes are applied
