# UNIXAI-Style Bash Megaliths

## Overview

A single, monolithic Bash script that turns the terminal into an intelligent, context-aware AI assistant. The reference implementation (UNIXAI Full Suite) integrates:

- Chat + reflection loops
- Agent command extraction (`<cmd>...</cmd>`)
- Firewall for dangerous commands
- Hardware automation (CPU governor / fan)
- Compiler-fix loop
- Config mutator
- Whisper transcription watchdog
- Web-scraping research agent
- IPC named-pipe listener
- Log-triage daemon

All of the above live inside one executable `.sh` file.

## Design Principles for Agentic Bash

1. **Self-contained** — minimal external dependencies; graceful fallbacks (jq → python3 → awk).
2. **Namespaced modules** — each major capability is a clearly delimited section with its own variables and command dispatcher (`cmd_firewall`, `cmd_automator`, …).
3. **Trap & cleanup hygiene** — EXIT / INT / TERM traps; skip-wait flags for clean exits.
4. **History as state** — bounded message history with character budget; trim under pressure.
5. **Tool tags as actions** — the model emits structured tags (`<cmd>`, `<fetch>`, `<extract>`); the harness parses and acts.
6. **Daemon orchestration** — background processes tracked in an associative array; start/stop/status commands.
7. **Safety first** — firewall intercepts risky patterns before execution; interactive confirmation.

## Structure Skeleton

```bash
#!/usr/bin/env bash
set -euo pipefail

# Global flags & configurable parameters
# Shared utilities (json_escape, extract_response, thinking spinner, …)
# Message history + prompt builder
# Model listing / selection
# MODULE: FIREWALL
# MODULE: AUTOMATOR
# MODULE: COMPILER FIX LOOP
# MODULE: CONFIG MUTATOR
# MODULE: WHISPER WATCHDOG
# MODULE: SCRAPE (web research agent)
# MODULE: IPC PIPE
# MODULE: TRIAGE DAEMON
# DAEMON MANAGEMENT
# BANNER & HELP
# MAIN loop (slash commands + conversation + reflection + command execution)
```

## Why This Fits Dynamic Programming

The entire script is the **value function** of the agentic session. Sub-problems (firewall rule matching, scrape turn, compiler iteration) are solved optimally and memoized in the running process state. An LLM can improve any module independently; re-running the single file reconstitutes the whole system.
