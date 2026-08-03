# 07 · Best Practices & Conventions

These conventions keep a megalith readable by both humans and future agents, even as it grows large.

---

## Header Discipline

Always include:

- Version (`X.Y.Z`)
- Architecture overview
- Known issues resolved (with symptom / location / root cause / fix / impact)
- Exact usage commands
- Generated file tree (or a pointer to it)

Keep the `LLM-INSTRUCTION` block more current than any other comment. Future agents will trust it more than the code itself.

---

## File Content Conventions

- Prefix important modules with a short `# LLM-INSTRUCTION: ...` comment.
- Prefer explicit, production-oriented defaults:
  - healthchecks that actually work
  - structured JSON logging
  - circuit breakers
  - trace IDs
- Keep Docker build contexts consistent (`context: .` plus correct `COPY` paths).
- Apply `sys.path` fixes uniformly in every service entry-point if the project uses a monorepo-style layout.

---

## Error Handling Inside the Megalith

When a bug is discovered, document it *before* applying the fix:

```
Known issues resolved in v1.2.0
--------------------------------
Symptom:  Gateway healthcheck fails with "ssl module not found"
Location: gateway/Dockerfile healthcheck CMD
Root cause: Used `python -c "import ssl"` but the slim image lacked the module
Exact fix: Changed to a curl-based healthcheck against /health
Impact:   Containers now pass healthchecks on first boot
```

Then apply the fix inside the corresponding `write_file` string and bump the version.

---

## Naming

| Item | Recommendation |
|------|----------------|
| Script | `bootstrap.py` (or `mspyb.py`, `megalith.py`) |
| Output directory | `project/` (configurable via `PROJECT_ROOT`) |
| Section separators | Full-width comment banners |

---

## Security & Production Notes

- Never hard-code real secrets. Always ship `.env.example`.
- Document CORS, rate limits, JWT secrets, etc. as configuration that *must* be tightened for production.
- Prefer self-signed certificate generation inside Dockerfiles for local HTTPS demos.
- Treat the megalith itself as potentially public; assume it will be shared with LLMs and agents.

---

## Size Management

For very large systems:

- Split logical sections into helper functions that return content strings:

  ```python
  def gateway_files():
      return {
          "gateway/Dockerfile": """...""",
          "gateway/app/main.py": """...""",
      }

  for path, content in gateway_files().items():
      write_file(path, content)
  ```

- Or keep multiple related MSPYB files and compose them.
- The ideal remains **one megalith per coherent platform**.

---

## Section Ordering (Recommended)

1. Root files (`.env.example`, `docker-compose.yml`, `README.md`, `requirements.txt` …)
2. Shared / core utilities
3. Infrastructure (databases, message queues, reverse proxies)
4. Gateway / edge services
5. Domain services (auth, billing, …)
6. Front-end or CLI entry points
7. Tests / fixtures (optional)

Consistent ordering lets both humans and agents locate any piece of the system quickly.

---

## What to Avoid

- Leaving outdated `LLM-INSTRUCTION` comments after an architecture change
- Writing files that depend on the existence of other files that are defined later in the same script (order can matter for clarity even if not for correctness)
- Embedding large binary assets; prefer generating them or downloading them at runtime
- Mixing generation logic with runtime logic inside the same strings

---

Next: concrete scenarios where the format delivers clear value in [Real-World Use Cases](08-use-cases.md).
