# Gemini 4 Hydrogen — Local Harness Chat Wrapper 

*by webXOS webxos.netlify.app // github.com/webxos*

A minimal and Single-file, stdlib-only Python harness that serves a chat UI on `127.0.0.1` and proxies prompts to the Gemini REST API.

## Requirements
1. Get a Gemini API Key (For free via https://aistudio.google.com/docs/whats-new-gemini-4-hydrogen)

2. Python 3.8+
3. `npx` / Node.js — **only** if you want the optional "Google OAuth" button (`@google/gemini-cli`)

## Run

```bash
cd /folder/ (The folder the py file is in)
python3 gemini_harness_by_webxos-v1.py             
```

If the requested port is busy, the harness falls back through `8420 → 8421 → 8422 → 8423`.

## Configure

1. Open the **Settings** drawer in the sidebar.
2. Paste your Gemini API key and pick a thinking level (`low` / `medium` / `high` / `xhigh`).
3. Save.

The key can also be seeded via `GEMINI_API_KEY` before first run. It is persisted in `config.json` next to the script and sent **only** to `generativelanguage.googleapis.com` via the `x-goog-api-key` header.

## Shell mode (`/sh`)

Prefix a prompt with `/sh` to ask the model for a single bash command, e.g.:

```
/sh show disk usage
/sh list large files in ~/Downloads
```

- **Read-only allowlist** (`uptime`, `df`, `ls`, `pwd`, …) runs immediately.
- **Everything else** is shown to you as a one-time confirmation token — click **Run** to execute, or **Discard**.
- **Deny list** (`rm`, `mkfs`, `shutdown`, `curl`, `sudo`, …) is refused even after confirmation.
- Multi-line commands and any shell metacharacter (`; | & $ > …`) never auto-run.

## Security notes

- Binds `127.0.0.1` only; strict `Origin` + `Host` checks on every request.
- Request bodies capped at 1 MiB; chunked and `Content-Length` framing validated.
- CSP + `X-Content-Type-Options: nosniff` on every response; model output rendered via `textContent`.
- Confirmation tokens: single-use, 10-minute TTL, FIFO-capped at 64.
- Rotating log at `harness.log` with the API key redacted everywhere.
- CLI auth: 30s timeout, process-group kill, single-flight guard.

## Files

| File            | Purpose                                    |
| --------------- | ------------------------------------------ |
| `geminiv3.py`   | The whole app (server + UI).               |
| `config.json`   | API key + thinking level (created on save).|
| `harness.log`   | Rotating request/execution log.            |

## Stop

`Ctrl+C`.

## Disclaimer

This is a local developer tool. Shell commands are not verified by Google. Only run commands you understand, and keep the listener on `127.0.0.1` — do not expose it to a network.

## LICENSE

MIT Open Sourced by github.com/webxos
