# REA_mini.py (Testing + Development) 
```
▄▖▄▖▄▖  ▖  ▖▄▖▖ ▖▄▖
▙▘▙▖▌▌  ▛▖▞▌▐ ▛▖▌▐ 
▌▌▙▖▛▌  ▌▝ ▌▟▖▌▝▌▟▖
```

A single-file, local-first Python harness for reverse-engineering shipped software artifacts. Inspired by [morluto/rea](https://github.com/morluto/rea). Free tools and free public APIs are prioritized at every step.

REA_mini.py bridges AI coding agents (via Model Context Protocol over stdio) and conventional reverse-engineering workflows. It captures URLs, queries free public APIs in parallel, drives Ghidra headless against local binaries, and generates runnable code skeletons in Python, JavaScript, HTML, Rust, or C++.

## Features

- **URL capture to Markdown** — Renders pages with Playwright, falls back to `urllib`, extracts the main article with `readability-lxml`, and converts it to Markdown with `html2text`. The result is optimized for LLM consumption.
- **Self-contained HTML archives** — Bundles a page and all its assets into a single file using `single-file-cli`.
- **Free API fan-out** — Queries GitHub, Wikipedia, Stack Exchange, Hacker News, PyPI, npm, crates.io, and GitHub code search in parallel, ranks results by keyword overlap with per-source weights, then diversifies with Maximal Marginal Relevance (MMR).
- **Ghidra headless analysis** — Drives `analyzeHeadless` against a local binary, dumps functions and printable ASCII strings via a bundled Java postScript, and scans them for keywords. Falls back to the `strings` utility when the postScript dump is unavailable.
- **Skeleton generation** — Asks a local or cloud LLM for a working program in Python, JavaScript, HTML, Rust, or C++, validates that it is syntactically correct, and falls back to a deterministic runnable template if the LLM output fails the quality gate.
- **Full pipeline** — Derives keywords from a URL, fans out across free APIs, and synthesizes a skeleton from the research context in a single command.
- **Dependency self-check** — Reports every missing dependency alongside its exact fix command.
- **Evidence sidecars** — Every artifact-producing command writes a `<artifact>.evidence.json` file recording the claim, evidence, limitations, unknowns, and reproducibility metadata.
- **Local-first LLM** — Requires a local Ollama model by default (`qwen2.5:0.5b`), auto-detects the daemon, starts it if necessary, and pulls the model if missing. Cloud providers are strictly opt-in and layered behind the local fallback.
- **MCP server** — Exposes the same tools over JSON-RPC 2.0 on stdio so AI coding agents can drive the harness.

---

## Requirements

- Python 3.9 or newer
- Optional, for capture to Markdown: `playwright`, `readability-lxml`, `html2text`, Chromium
- Optional, for archive mode: Node.js and `npx`
- Optional, for Ghidra analysis: Java 21 or newer, `binutils` (`strings`)
- Optional, for skeleton generation: a local Ollama daemon, or a cloud LLM API key

The script always loads and runs even when optional dependencies are missing. Individual commands fail with a clear message identifying the missing dependency.

---

## Installation

Download the rea_mini.py file and place it in any folder on your system. Use at your own risk.

```bash
cd /(folder you put the file in)/
python3 rea_mini.py 
```

On first interactive run, REA_mini.py checks every dependency and offers a single `[A]ll / [N]one / [I]nteractive` prompt so you do not answer ten questions in a row.

---

## Quick Start

```bash
# 1. Verify your environment
python rea_mini.py --cmd selfcheck

# 2. Capture a page as Markdown
python rea_mini.py --cmd capture --url https://example.com

# 3. Search across free APIs
python rea_mini.py --cmd fanout --query "rust async runtime"

# 4. Generate a runnable Python skeleton from an idea
python rea_mini.py --cmd generate --lang python \
    --idea "slugify a URL into a filesystem-safe filename"

# 5. Full pipeline: URL -> fan-out -> skeleton
python rea_mini.py --cmd full --url https://github.com/some/repo --lang rust
```

All artifacts are written to `./rea_output/` by default. Override with `--output <dir>` or `-o <dir>`.

---

## Interactive Mode

Run without arguments on a terminal:

```bash
python rea_mini.py
```

The startup wizard does the following:

1. Verifies that a local Ollama model is installed, pulls it if needed, and smoke-tests it.
2. Audits optional dependencies and prompts for installation.
3. Offers a one-time `EXPAND LLM?` prompt to add an optional cloud provider. The decision is remembered.
4. Prints a status banner showing local model, cloud provider, and Ghidra availability.

After onboarding, the menu appears:

```
============================================================
 REA_mini.py — reverse-engineer-anything mini harness
============================================================
 LLM: local:Y (qwen2.5:0.5b) cloud:-
  1. Capture URL -> clean Markdown
  2. Capture URL -> self-contained HTML archive
  3. Search fan-out (free public APIs)
  4. Ghidra headless analysis (binary + keywords)
  5. Generate skeleton file (HTML/JS/Py/Rust/C++)
  6. Full pipeline (URL -> fan-out -> synth -> skeleton)
  7. Self-check dependencies
  0. Quit
============================================================
```

Ctrl+C on any menu action returns to the menu without aborting the process.

---

## Non-Interactive CLI Mode

Every menu action has a `--cmd` equivalent suitable for scripts and CI.

```bash
python rea_mini.py --cmd capture --url <url>               # Markdown
python rea_mini.py --cmd archive --url <url>               # Self-contained HTML
python rea_mini.py --cmd fanout  --query <terms>           # Free API search
python rea_mini.py --cmd ghidra  --binary <path> --keywords foo,bar
python rea_mini.py --cmd generate --lang python --idea "<spec>"
python rea_mini.py --cmd full    --url <url> --lang rust
python rea_mini.py --cmd selfcheck
python rea_mini.py --cmd mcp                               # stdio MCP server
python rea_mini.py --cmd validate-evidence --evidence-file <path>
python rea_mini.py --list-models                           # LLM config summary
```

Non-interactive `--cmd` invocations never run the onboarding wizard. Commands that require an LLM fail fast with exit code `10` unless a cloud provider is configured or `--no-require-local-llm` is passed.

---

## Command Reference

### `capture`

Fetch a URL and extract its main content as Markdown.

```
--url <url>                Required.
--mode llm|archive         Defaults to llm.
--scroll                   Scroll to page bottom before extraction.
--wait-selector <css>      Wait for a CSS selector before extraction.
--timeout <seconds>        Override the default 30s navigation timeout.
```

Output: `rea_output/<host>_<path>_llm_<timestamp>.md` plus an evidence sidecar.

### `archive`

Bundle a URL into a single self-contained HTML file via `single-file-cli`.

```
--url <url>                Required.
```

Output: `rea_output/<host>_<path>_archive_<timestamp>.html` plus an evidence sidecar.

### `fanout`

Query all free public APIs in parallel.

```
--query <terms>            Required.
```

Ranked, MMR-diversified results are printed to stdout and saved as JSON:

```
rea_output/fanout_<host>_<path>_fanout_<timestamp>.json
```

The cache lives in `rea_output/.cache/fanout_<hash>.json` with a one-hour TTL.

### `ghidra`

Run Ghidra headless against a local binary.

```
--binary <path>            Required.
--keywords foo,bar         Optional, comma-separated.
```

If Ghidra is not present, you are offered a download prompt (approximately 400 MB). The SHA-256 of the downloaded archive is verified against the published digest when available.

Output: `rea_output/ghidra_<binary>_<timestamp>.json` containing function list, printable strings, keyword hits, and analyzer metadata.

### `generate`

Generate a runnable skeleton in Python, JavaScript, HTML, Rust, or C++.

```
--lang python|js|html|rust|cpp   Required.
--idea "<spec>"                  Required.
--run-generated                  Optional. Execute the artifact's --test mode
                                 after generation (see warning below).
```

Default behavior:

1. Ask the LLM for a complete program in a single fenced code block.
2. If the LLM output does not start with a valid token for the language (import, function, `<!doctype`, etc.), fall back to the deterministic template.
3. Write the artifact and run a safe syntax check (`py_compile`, `node --check`, `rustc --emit=metadata`, `g++ -fsyntax-only`, HTML structural check).
4. If the syntax check fails and the provider was an LLM, rewrite the artifact with the deterministic template.

Warning: `--run-generated` executes LLM-generated code. Only use it with trusted models and sandboxed environments. Without this flag, generated code is only syntax-checked, never executed.

Output: `rea_output/skel_<uuid>_<idea>.<ext>` plus an evidence sidecar.

### `full`

Derive keywords from a URL, fan out across free APIs, and synthesize a skeleton from the research context.

```
--url <url>                Required.
--lang python|js|html|rust|cpp   Required.
--run-generated            Optional. Same caveat as above.
```

### `selfcheck`

Report installed dependencies and print exact fix commands for missing ones.

```
--yes                      Auto-install missing pip packages without prompting.
```

### `mcp`

Start a JSON-RPC 2.0 Model Context Protocol server on stdio. See [MCP Server](#mcp-server).

### `validate-evidence`

Validate that an evidence sidecar matches the expected schema.

```
--evidence-file <path>     Required.
```

Returns exit code `0` when valid, `9` when invalid.

---

## MCP Server

Start the server:

```bash
python rea_mini.py --cmd mcp
```

The server reads newline-delimited JSON-RPC 2.0 on stdin and writes responses to stdout. Stdin reads run off the event loop, so long-running tool calls do not block receipt of the next request. Each tool call has a 300-second timeout.

Exposed tools:

| Tool | Arguments | Description |
|---|---|---|
| `rea_capture_markdown` | `url` | Capture a URL and return Markdown |
| `rea_fanout` | `query` | Free public API fan-out, returns JSON |
| `rea_ghidra_analyze` | `binary`, `keywords` | Ghidra headless analysis, returns JSON |
| `rea_generate_skeleton` | `idea`, `lang` | LLM or template skeleton |
| `rea_full_pipeline` | `url`, `lang` | URL to skeleton |
| `rea_selfcheck` | — | Dependency report |

Wire the server into your agent by pointing it at:

```json
{
  "command": "python",
  "args": ["/path/to/rea_mini.py", "--cmd", "mcp"]
}
```

---

## LLM Configuration

### Local-first (default)

On interactive startup, the harness ensures a local Ollama model is available. The default starter model is `qwen2.5:0.5b`. You can override it:

```bash
export REA_LLM_OLLAMA_MODEL=llama3.2:3b
```

The resolved model is persisted in `rea_mini_config.json` and `ollama_cache.json`.

### Optional cloud expansion

After the local model works, you are prompted once: `EXPAND LLM? Would you like to also add a cloud model? [y/N]`. Declining is remembered. Enabling a provider asks for its API key, validates it with a smoke test, and layers it behind the local fallback.

Supported cloud providers:

| Provider | Env var | Default model |
|---|---|---|
| Groq | `GROQ_API_KEY` | `llama-3.1-8b-instant` |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-3.1-8b-instruct:free` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-20241022` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest` |
| Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| Hugging Face | `HF_TOKEN` | `HuggingFaceH4/zephyr-7b-beta` |

### Skipping the local requirement

For CI environments without Ollama:

```bash
python rea_mini.py --cmd generate --lang python --idea "..." --no-require-local-llm
```

In this mode, REA_mini.py tries every configured cloud provider in order, then Ollama last, then falls back to deterministic templates.

### Inspecting resolution

```bash
python rea_mini.py --list-models
```

Prints the required-local flag, resolved models, per-provider readiness (with env/config override indicators), and the current `OLLAMA_HOST`.

---

## Evidence Sidecars

Every artifact-producing command writes a sibling `<artifact>.evidence.json` file. This is the same schema agents use to reason about what was actually produced.

```json
{
  "schema_version": "1.1",
  "claim": "Extracted LLM-friendly Markdown from https://example.com",
  "evidence": [
    {
      "artifact": "rea_output/example_com_index_llm_20260101-120000.md",
      "source_url": "https://example.com",
      "final_url": "https://example.com",
      "bytes": 12345,
      "tool": "playwright+readability+html2text",
      "used_playwright": true
    }
  ],
  "limitations": ["Heavy SPAs, paywalls, and JS-only content may be truncated or missing."],
  "unknowns": ["Whether the rendered DOM matched the version a logged-in user would see."],
  "extra": {},
  "reproducibility": {
    "argv": ["rea_mini.py", "--cmd", "capture", "--url", "https://example.com"],
    "python_version": "3.11.5",
    "python_executable": "/usr/bin/python3",
    "platform": "Linux x86_64",
    "cwd": "/home/user/projects",
    "optional_deps": {"playwright": true, "readability_lxml": true, "html2text": true}
  },
  "generated_at": "2026-01-01T12:00:00.000000+00:00"
}
```

Validate an existing sidecar:

```bash
python rea_mini.py --cmd validate-evidence --evidence-file rea_output/foo.md.evidence.json
```

---

## Configuration File

`rea_mini_config.json` is created on first interactive run. Example:

```json
{
  "mode": "llm",
  "timeout": 30,
  "output_dir": "rea_output",
  "ignore_images": false,
  "http_timeout": 20,
  "fanout_cache_ttl_s": 3600,
  "archive_cli_version": "2.0.75",
  "scroll_to_bottom": false,
  "wait_selector": "",
  "mmr_lambda": 0.7,
  "mmr_cap": 100,
  "archive_timeout_s": 600,
  "require_local_llm": true,
  "llm_fallback_model": "qwen2.5:0.5b",
  "cloud_provider": "",
  "cloud_model": "",
  "strict_local_llm_skip_cloud": true
}
```

The following caches are also persisted next to the script:

- `ghidra_cache.json` — Cached Ghidra path, downloaded release tag, SHA-256, and whether `-analysisTimeoutPerFile` is supported.
- `ollama_cache.json` — Resolved local model, host, and cloud opt-in decision.
- `rea_mini_error.log` — Rotated at 5 MB.

CLI flags always override the config file for the current run. `--output` sets the artifact directory. `--timeout`, `--scroll`, `--wait-selector`, and `--no-require-local-llm` are applied before dispatch.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `OLLAMA_HOST` | Override the Ollama daemon URL (default `http://localhost:11434`) |
| `REA_LLM_OLLAMA_MODEL` | Pin the local Ollama model |
| `REA_LLM_<PROVIDER>_MODEL` | Pin a cloud provider's model (`GROQ`, `OPENROUTER`, `OPENAI`, `ANTHROPIC`, `DEEPSEEK`, `MISTRAL`, `GEMINI`, `HF`) |
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN` | Cloud provider credentials |
| `GITHUB_TOKEN` | Enables GitHub code search; also raises GitHub API rate limits |
| `GHIDRA_HOME` | Explicit Ghidra installation directory |

Model env vars take precedence over config-file overrides. Config overrides take precedence over built-in defaults.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | Python older than 3.9 |
| `3` | Missing dependencies |
| `4` | Bad URL or input |
| `5` | Navigation failed |
| `6` | Extraction returned empty |
| `7` | Write failed |
| `8` | Archive failed |
| `9` | Invalid evidence file |
| `10` | Local LLM required but unavailable |
| `130` | Interrupted (Ctrl+C) |

Non-interactive `--cmd` invocations of LLM-requiring commands return `10` when `require_local_llm` is set and no LLM is reachable, rather than silently producing a template.

---

## Troubleshooting

### "Ollama is not installed"

Install from [ollama.com/download](https://ollama.com/download). Or, on Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then re-run REA_mini.py. If the daemon is not running, the harness attempts to start it automatically.

### "Playwright path failed; falling back to static fetch"

The static fallback uses `urllib` plus `readability-lxml` and works for most content. If the target site is a JavaScript-heavy SPA and the fallback is truncating it, install Chromium:

```bash
python -m playwright install chromium
sudo apt install libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2
```

### "Java 21+ required for Ghidra"

```bash
sudo apt install openjdk-21-jdk
```

The Ghidra download prompt appears automatically if it is not found. You can also set `GHIDRA_HOME` to an existing installation.

### "strings" (binutils) not found

```bash
sudo apt install binutils
```

The Ghidra keyword scan falls back to the postScript dump when `strings` is unavailable. This is often sufficient.

### LLM output rejected by the quality gate

The message `(LLM output rejected by quality gate; used deterministic template.)` means the LLM's output did not begin with a valid token for the requested language or failed its syntax check. The deterministic template is written instead. To investigate:

- Set `--run-generated` on a trusted model to see the smoke-test failure details in the evidence sidecar.
- Check `rea_mini_error.log` for `llm_output_rejected` and `skeleton_quality_gate_failed` entries.
- Consider a larger model via `REA_LLM_OLLAMA_MODEL`.

### Rate limits on GitHub

Set `GITHUB_TOKEN` to raise the search and code-search limits from 10 and 60 requests/hour respectively.

### Fan-out returns no results for a query

Free public APIs sometimes return empty result sets for narrow queries. The cache at `rea_output/.cache/fanout_<hash>.json` has a one-hour TTL. To force a refetch, delete the file or adjust `fanout_cache_ttl_s` in `rea_mini_config.json`.

---

## Design Notes

- **Local-first.** A local Ollama model is required by default. Cloud providers are opt-in and layered behind it. `--no-require-local-llm` exists specifically for CI.
- **Evidence over assertion.** Every artifact-producing command writes a sidecar. The schema is stable and validated by `--cmd validate-evidence`. `used_playwright` is tracked explicitly to prevent false "Playwright was used" reports.
- **Deterministic fallbacks.** Skeleton generation, dependency installation, and Ghidra retrieval all degrade gracefully. The harness never aborts on a missing optional feature; it reports and continues.
- **Safe-by-default code handling.** Generated artifacts are syntax-checked only. The heavier smoke test (executing the artifact's `--test` mode) is gated behind `--run-generated` and documented as executing LLM-generated code.
- **MMR diversification.** Fan-out applies per-source keyword weighting, then MMR with `lambda=0.7` over the first 100 hits. The tail is preserved unchanged. Complexity is `O(cap^3)` in the head; at `cap=100` this is roughly one million operations, well within budget.
- **MCP-friendly.** The server reads stdin off the event loop so long-running Ghidra or capture calls do not wedge the harness from receiving the next request. Each tool call has a 300-second timeout.
- **Cached capabilities.** Whether `analyzeHeadless` supports `-analysisTimeoutPerFile` is probed once and persisted in `ghidra_cache.json`. This avoids adding a one-second subprocess to every Ghidra invocation.
- **ZIP-slip protection.** Ghidra's ZIP is extracted with per-entry path traversal checks. The published SHA-256 digest is verified when available.

---

## Attribution

Inspired by [morluto/rea](https://github.com/morluto/rea) and the broader reverse-engineer-anything project. This file is a self-contained, single-file implementation intended for local use. See the upstream project for the full-featured MCP server and skill definitions.
