#!/usr/bin/env python3
"""Gemini 4 Hydrogen // Minimal Harness (v3, production-grade).

A single-file, stdlib-only local harness serving a DeepSeek-style
monochrome chat UI on 127.0.0.1 only.

Security model (v3 hardening, all JGF issues resolved):
  - Binds 127.0.0.1; strict Origin + Host validation on every request
    (CSRF / DNS-rebinding defense).                      [v1-C1, v1-H1]
  - /sh requests: read-only allowlist auto-runs; anything else needs
    explicit confirmation via a one-time token; deny list refused even
    after confirmation; multi-line commands NEVER auto-run. [v1-C2, v1-C3, v2-C1]
  - Pending confirmation tokens are FIFO-capped, expiring, single-use.
                                                        [v2-H1]
  - Request bodies capped (413) for both Content-Length and chunked
    framing; invalid Content-Length is rejected 400, not silently
    dropped.                                            [v1-H2, v2-H2, v2-M1]
  - Config: RLock + atomic replace + strict validation; invalid values
    raise 4xx back to the caller instead of silently vanishing.
                                                        [v1-H3, v1-M4, v2-M2]
  - Gemini API key travels in the x-goog-api-key header, never the URL.
                                                        [v1-M3]
  - urlopen has a 60s timeout.                          [v1-H4]
  - CLI auth: 30s timeout, process-group kill, single-flight guard
    (no concurrent npx spawns).                        [v1-H5, v1-L3]
  - CSP + X-Content-Type-Options: nosniff on every response; the page
    never injects model output as HTML.                 [v2-M5, 4.7]
  - Rotating file log; API key redacted everywhere it may appear.
                                                        [v1-L1, v2-L4]
  - Execution output truncated; allowlist auto-runs are logged.
                                                        [v1-L2, v2-L3]
  - Port via --port / HARNESS_PORT with fallback chain. [v1/n9]

Run: python3 harness.py [--port N] [--no-browser]
"""

import argparse
import http.server
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import secrets
import signal
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
LOG_FILE = os.path.join(APP_DIR, "harness.log")

DEFAULT_PORT = 8420
FALLBACK_PORTS = [8420, 8421, 8422, 8423]
HOST = "127.0.0.1"
API_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-4-hydrogen:generateContent"
)
GEMINI_API_TIMEOUT = 60
CLI_AUTH_TIMEOUT = 30          # npx may need to download the package first
MAX_BODY_BYTES = 1024 * 1024   # 1 MiB cap on request bodies
MAX_OUTPUT_CHARS = 20000       # truncate huge command output in replies
EXEC_TIMEOUT = 60              # hard cap for confirmed shell executions
VALID_THINKING = ("low", "medium", "high", "xhigh")
PENDING_CAP = 64               # v2-H1: hard cap on unconfirmed commands
PENDING_TTL = 600               # unconfirmed commands expire after 10 min
CSP_POLICY = (
    "default-src 'self'; script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; connect-src 'self'"
)

# v1-C3/v2-L1: read-only commands safe to auto-run WITHOUT confirmation.
# Matched against a NORMALIZED SINGLE LINE only (v2-C1: any newline or
# shell metacharacter disqualifies auto-run entirely).
AUTO_SAFE_COMMANDS = (
    "uptime", "date", "whoami", "id", "hostname", "uname", "df", "free",
    "ls", "pwd", "cat /etc/os-release", "echo",
)
# Absolute blocks: word-boundary regex, so "format" no longer matches
# inside "information" (v2-L1). Never executed, confirmed or not.
DENIED_PATTERNS = (
    r"\brm\b", r"\bmkfs\b", r"\bdd\b", r"\bshutdown\b", r"\breboot\b",
    r"\bkill\b", r"\bpkill\b", r"\bchmod\b", r"\bchown\b", r"\bcurl\b",
    r"\bwget\b", r"\bnc\b", r"\bssh\b", r"\bscp\b", r"\bformat\b",
    r"\bpasswd\b", r"\buseradd\b", r"\buserdel\b", r"\bvisudo\b",
    r"\bsudo\b", r"\beval\b", r"\bsource\b", r"\bexec\b", r"\bfork\b",
    r"/dev/sd", r"/dev/nvme", r":\(\)\{", r"\bhistory\b", r"\bmv\b",
    r"\bcp\b", r"\btouch\b", r"\bmkdir\b", r"\brmdir\b", r"\btee\b",
    r"\bcrontab\b", r"\biptables\b", r"\bnpm\b", r"\bpip\b",
    r"\bgit\b", r"\bgrep\b", r"\bsed\b", r"\bawk\b", r"\btee\b",
    r"\benv\b", r"\bexport\b", r"\bset\b", r"\bunset\b", r">\s*/dev/",
)
DENIED_RES = [re.compile(p) for p in DENIED_PATTERNS]

# v1-M2: model replies that are refusals or prose, never shell commands.
REFUSAL_PATTERNS = (
    "i cannot", "i can't", "i'm sorry", "i am sorry", "i won't",
    "i will not", "cannot assist", "can't assist", "cannot help",
    "can't help", "not able to", "unable to", "against my", "sorry, but",
    "as an ai", "as a language model", "i must decline", "must decline",
    "cannot provide", "can't provide", "i do not have", "i don't have",
    "i apologize", "apologize, but",
)

# v1-M1: fenced code blocks (```bash ... ``` etc.) and inline code spans.
RE_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*\s*\n)?(.*?)```", re.DOTALL)
RE_INLINE = re.compile(r"`([^`\n]+)`")
# Any shell metacharacter disqualifies auto-run (v2-C1 hardening).
SHELL_METACHARS = (";", "|", "&", "(", ")", "$", "<", ">", "`", "\\", "\n", "\r")


# --------------------------------------------------------------------------
# Logging (v1-L1): requests go to a rotating harness.log.
# --------------------------------------------------------------------------
def _build_logger():
    logger = logging.getLogger("harness")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        try:
            fh = RotatingFileHandler(
                LOG_FILE, maxBytes=512 * 1024, backupCount=2, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass  # read-only FS: keep serving, log to memory only
    return logger


LOG = _build_logger()


def _redact(text):
    """v2-L4: strip the API key from log/output strings.

    Catches both `key=...` fragments and the literal configured key
    value, whatever its position, so the key can never reach the log.
    """
    if not isinstance(text, str) or not text:
        return text
    redacted = re.sub(r"key=[A-Za-z0-9_\-]+", "key=[REDACTED]", text)
    try:
        cfg_key = load_cfg().get("apiKey") or ""
    except Exception:
        cfg_key = ""
    if cfg_key and len(cfg_key) >= 8:
        redacted = redacted.replace(cfg_key, "[REDACTED]")
    return redacted


# --------------------------------------------------------------------------
# Config (v1-H3, v1-M4, v2-M2, v2-M3): one RLock, atomic writes, strict
# validation; bad input raises ValueError -> HTTP 400 (no silent drop,
# no dead cache variable).
# --------------------------------------------------------------------------
_CFG_LOCK = threading.RLock()


def _cfg_defaults():
    return {"apiKey": os.environ.get("GEMINI_API_KEY", ""), "thinking": "high"}


def _validate_cfg_value(key, value):
    """Normalize + validate one config value; raises ValueError."""
    if key == "apiKey":
        if not isinstance(value, str):
            raise ValueError("apiKey must be a string")
        if value and len(value) > 512:
            raise ValueError("apiKey too long (max 512 chars)")
        return value
    if key == "thinking":
        if not isinstance(value, str):
            raise ValueError("thinking must be a string")
        v = value.strip().lower()
        if v not in VALID_THINKING:
            raise ValueError(f"thinking must be one of {list(VALID_THINKING)}")
        return v
    raise ValueError(f"unknown config key {key!r}")


def load_cfg():
    """Load config.json under lock; corrupt files fall back to defaults."""
    with _CFG_LOCK:
        cfg = _cfg_defaults()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    for k in ("apiKey", "thinking"):
                        if k in saved and isinstance(saved[k], str):
                            try:
                                cfg[k] = _validate_cfg_value(k, saved[k])
                            except ValueError:
                                LOG.warning("config.json value %s invalid; using default", k)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                LOG.warning("config.json unreadable; using defaults")
        return dict(cfg)


def save_cfg(update):
    """Validate + merge `update` into config.json atomically.

    Raises ValueError on any bad key/value (v2-M2), so the HTTP layer can
    answer 400; a failed save never wipes existing data (tmp + replace).
    """
    if not isinstance(update, dict):
        raise ValueError("config update must be a JSON object")
    with _CFG_LOCK:
        cfg = load_cfg()
        clean = {}
        for k, v in update.items():
            clean[k] = _validate_cfg_value(k, v)   # raises on bad input
        cfg.update(clean)
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CONFIG_FILE)
        return dict(cfg)


# --------------------------------------------------------------------------
# Pending-execution tokens (v1-C2, v2-H1): one-time, expiring, FIFO-cap.
# --------------------------------------------------------------------------
_PENDING_LOCK = threading.Lock()
_PENDING = {}          # token -> {"cmd", "prompt", "expires", "seq"}
_PENDING_SEQ = [0]     # monotonic insertion counter for FIFO eviction


def _register_pending(cmd, prompt):
    """Register a command awaiting confirmation; FIFO-capped (v2-H1)."""
    token = secrets.token_urlsafe(24)
    with _PENDING_LOCK:
        now = time.time()
        # drop expired entries first
        for t in [t for t, v in _PENDING.items() if v["expires"] < now]:
            del _PENDING[t]
        # hard cap: evict the OLDEST entry (FIFO), never grow unbounded
        while len(_PENDING) >= PENDING_CAP:
            oldest = min(_PENDING, key=lambda t: _PENDING[t]["seq"])
            del _PENDING[oldest]
        _PENDING_SEQ[0] += 1
        _PENDING[token] = {
            "cmd": cmd,
            "prompt": prompt,
            "expires": now + PENDING_TTL,
            "seq": _PENDING_SEQ[0],
        }
    return token


def _pop_pending(token):
    """Return the command for a valid, unexpired token; consume it."""
    if not isinstance(token, str) or not (1 <= len(token) <= 128):
        return None
    with _PENDING_LOCK:
        item = _PENDING.pop(token, None)
    if item is None or item["expires"] < time.time():
        return None
    return item["cmd"]


def pending_count():
    with _PENDING_LOCK:
        return len(_PENDING)


# --------------------------------------------------------------------------
# Command parsing / classification (v1-M1, v1-M2, v2-C1, v2-L1)
# --------------------------------------------------------------------------
def extract_command(reply):
    """v1-M1: pull the shell command out of a model reply.

    Order: fenced block (```bash / ```sh / ```), then inline backtick
    span, then the raw text if it is command-shaped. Returns None when
    nothing command-like is found.
    """
    if not reply:
        return None
    text = reply.strip()
    if not text:
        return None
    m = RE_FENCE.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    m = RE_INLINE.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    return text


def looks_like_command(text):
    """v1-M2: heuristic -- does this text plausibly contain a shell command?"""
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0]
    if len(first) > 300:
        return False
    if any(p in first for p in ("?", "!", "; but", " however", " instead,")):
        return False
    if re.match(r"^[A-Z][a-z]+( [a-z]+){2,}$", first) and " " in first.strip("."):
        return False
    head = first.split()[0]
    if not re.match(r"^[A-Za-z0-9_\-./]+$", head):
        return False
    return True


def is_refusal(reply):
    """v1-M2: True if the model refused or answered in prose."""
    low = reply.lower()
    return any(p in low for p in REFUSAL_PATTERNS)


def command_risk(cmd):
    """Classify: 'safe' (auto-run), 'blocked' (never), 'confirm'.

    v2-C1: multi-line commands (newline/CR anywhere in the ORIGINAL
    string) are classified BEFORE any flattening, and can never be
    'safe'. v2-L1: deny patterns are word-boundary regexes on the
    flattened text.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return "blocked"
    # v2-C1: original (un-flattened) newline check comes FIRST.
    if "\n" in cmd or "\r" in cmd:
        flat = " ".join(cmd.split())
        if any(rx.search(flat) for rx in DENIED_RES):
            return "blocked"
        return "confirm"
    flat = " ".join(cmd.split())
    if any(rx.search(flat) for rx in DENIED_RES):
        return "blocked"
    # any shell metacharacter disqualifies auto-run
    if any(ch in flat for ch in SHELL_METACHARS):
        return "confirm"
    low = flat.lower()
    for p in AUTO_SAFE_COMMANDS:
        if low == p or low.startswith(p + " ") or low == p + " ":
            return "safe"
    return "confirm"


def build_system_prompt(shell_mode):
    base = (
        "You are Gemini 4 Hydrogen running in a minimal local Unix harness. "
        "Keep responses concise."
    )
    if shell_mode:
        base += (
            " Output ONLY one raw bash command to accomplish the user's request. "
            "No markdown fences, no explanation, no prose, no newlines. Prefer "
            "read-only commands (ls, df, ps, uptime)."
        )
    return base


def split_prompt(prompt):
    """v1-M1: detect the /sh prefix; returns (shell_mode, user_text)."""
    if prompt == "/sh" or prompt.startswith("/sh "):
        return True, prompt[3:].strip()
    return False, prompt


# --------------------------------------------------------------------------
# UI: DeepSeek-style full-viewport monochrome chat. Guide 4.1-4.8.
# ASCII only; black/white tokens; Times New Roman chrome, Courier code;
# invert via CSS variables (no CSS filter inversion); localStorage persistence
# applied pre-paint; all dynamic text via textContent (XSS-safe).
# --------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'">
<title>Gemini 4 Hydrogen Harness</title>
<script>
(function () {
  // 4.5: apply saved theme before first paint (no flash).
  try {
    var saved = localStorage.getItem("harness-theme");
    var dark = saved ? saved === "dark"
      : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
    if (dark) { document.documentElement.classList.add("dark"); }
  } catch (e) { /* storage unavailable: default light */ }
})();
</script>
<style>
  :root {
    --bg: #ffffff; --fg: #000000; --muted: #666666;
    --line: #000000; --surface: #f2f2f2;
    --bubble-user: #e8e8e8; --bubble-ai: #ffffff;
  }
  html.dark {
    --bg: #000000; --fg: #ffffff; --muted: #999999;
    --line: #ffffff; --surface: #1a1a1a;
    --bubble-user: #2e2e2e; --bubble-ai: #000000;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; overflow: hidden; }
  body {
    background: var(--bg); color: var(--fg);
    font-family: "Times New Roman", Times, serif; font-size: 16px;
  }
  button, input, select, textarea {
    font-family: "Times New Roman", Times, serif; font-size: 15px;
    background: var(--bg); color: var(--fg);
    border: 1px solid var(--line); padding: 5px 10px; cursor: pointer;
  }
  button:focus, input:focus, select:focus, textarea:focus {
    outline: 2px solid var(--fg); outline-offset: 1px;
  }
  button:hover { background: var(--surface); }

  /* 4.1 layout skeleton */
  .app { display: flex; height: 100vh; width: 100vw; }
  .sidebar {
    width: 260px; flex: 0 0 260px; display: flex; flex-direction: column;
    border-right: 1px solid var(--line); overflow-y: auto; background: var(--bg);
  }
  .sidebar-head { padding: 14px 14px 8px; border-bottom: 1px solid var(--line); }
  .sidebar-title { font-size: 18px; font-weight: bold; margin: 0; }
  .sidebar-sub { font-size: 12px; color: var(--muted); margin: 2px 0 0; }
  .newchat { width: 100%; margin: 10px 14px 0 0; text-align: left; }
  .convs { flex: 1; padding: 8px 14px; overflow-y: auto; }
  .conv { padding: 7px 8px; border: 1px solid transparent; margin-bottom: 5px; cursor: pointer; }
  .conv:hover, .conv.active { border-color: var(--line); background: var(--surface); }
  .conv-title { font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conv-preview { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sidebar-foot { border-top: 1px solid var(--line); padding: 10px 14px; }
  .foot-row { display: flex; gap: 8px; flex-wrap: wrap; }

  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 14px; border-bottom: 1px solid var(--line); gap: 10px;
  }
  .model-name { font-size: 15px; font-weight: bold; }
  .hamburger { display: none; }

  .scroller { flex: 1; overflow-y: auto; padding: 18px 22px 10px; }
  .msg-row { display: flex; margin-bottom: 14px; }
  .msg-row.user { justify-content: flex-end; }
  .msg-row.ai { justify-content: flex-start; }
  /* 4.2: user right bubble, assistant left plain block */
  .bubble-user {
    max-width: 72%; background: var(--bubble-user);
    border: 1px solid var(--line); border-radius: 10px;
    padding: 9px 13px; white-space: pre-wrap; word-break: break-word;
  }
  .bubble-ai {
    width: 100%; background: var(--bubble-ai);
    border: 1px solid var(--line); border-radius: 10px;
    padding: 10px 14px; white-space: pre-wrap; word-break: break-word;
  }
  .bubble-ai pre, .cmd-out {
    font-family: "Courier New", Courier, monospace; font-size: 13px;
    background: var(--surface); border: 1px solid var(--line);
    padding: 9px; white-space: pre-wrap; word-break: break-all;
    margin: 8px 0 2px;
  }
  .msg-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .msg-actions { margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap; }
  .msg-actions button { font-size: 13px; padding: 3px 10px; }
  .typing { color: var(--muted); font-style: italic; padding: 6px 0; }
  .sysmsg { color: var(--muted); font-size: 13px; border-left: 3px solid var(--line); padding: 4px 10px; margin-bottom: 14px; }
  .sysmsg a { color: var(--fg); text-decoration: underline; }
  .a11y-vis { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }

  /* 4.3 composer */
  .composer {
    border-top: 1px solid var(--line); padding: 10px 22px 6px; background: var(--bg);
  }
  .comp-row { display: flex; gap: 10px; align-items: flex-end; }
  .comp-box { flex: 1; min-width: 0; position: relative; }
  .hint-chip {
    display: none; font-size: 12px; color: var(--muted);
    border: 1px solid var(--line); padding: 1px 8px; margin-bottom: 4px;
    font-family: "Courier New", Courier, monospace;
  }
  .hint-chip.on { display: inline-block; }
  textarea#prompt {
    width: 100%; resize: none; min-height: 44px; max-height: 132px;
    line-height: 1.35; overflow-y: auto; display: block;
  }
  .send { flex: 0 0 auto; min-width: 84px; }
  .send[disabled] { opacity: 0.55; cursor: default; }
  .disclaimer { font-size: 12px; color: var(--muted); margin: 5px 0 2px; }

  /* 4.7 settings drawer (collapsible) */
  .drawer {
    display: none; border-bottom: 1px solid var(--line);
    padding: 10px 14px; background: var(--surface);
  }
  .drawer.open { display: block; }
  .drawer label { font-size: 13px; display: block; margin-top: 8px; }
  .drawer input[type="password"], .drawer select {
    width: 100%; margin-top: 3px; background: var(--bg);
  }
  .drawer .note { font-size: 12px; color: var(--muted); margin: 6px 0 0; }

  /* 4.1 mobile: off-canvas sidebar drawer */
  @media (max-width: 860px) {
    .sidebar {
      position: fixed; top: 0; left: 0; bottom: 0; z-index: 30;
      transform: translateX(-100%); transition: transform 0.18s ease;
      width: 82vw; max-width: 300px; background: var(--bg);
      box-shadow: 4px 0 0 0 var(--line);
    }
    .sidebar.open { transform: translateX(0); }
    .hamburger { display: inline-block; }
    .scroller { padding: 14px 12px 8px; }
    .composer { padding: 8px 12px 4px; }
    .bubble-user { max-width: 88%; }
    .backdrop {
      display: none; position: fixed; inset: 0; z-index: 25;
      background: var(--muted); opacity: 0.55;
    }
    .backdrop.on { display: block; }
  }
  .backdrop { display: none; }
</style>
</head>
<body>
<div class="app">

  <aside class="sidebar" id="sidebar" aria-label="Chat list">
    <div class="sidebar-head">
      <p class="sidebar-title">Hydrogen Harness</p>
      <p class="sidebar-sub">gemini-4-hydrogen / local</p>
    </div>
    <div style="padding:10px 14px 0;">
      <button type="button" id="newchat" class="newchat" style="width:100%;">New chat</button>
    </div>
    <div class="convs" id="convs"></div>
    <div class="sidebar-foot">
      <div class="foot-row">
        <button type="button" id="settings-btn">Settings</button>
        <button type="button" id="invert-side">Invert</button>
        <button type="button" id="auth-btn">Google OAuth</button>
      </div>
      <div class="drawer" id="drawer">
        <form onsubmit="return false">
          <label for="apiKey">API key</label>
          <input type="password" id="apiKey" placeholder="GEMINI_API_KEY" autocomplete="off">
          <label for="thinking">Thinking level</label>
          <select id="thinking">
            <option value="low">low (fastest)</option>
            <option value="medium">medium (balanced)</option>
            <option value="high" selected>high (default)</option>
            <option value="xhigh">xhigh (deep reasoning)</option>
          </select>
          <div class="foot-row" style="margin-top:10px;">
            <button type="button" id="save-cfg">Save</button>
            <button type="button" id="cancel-cfg">Close</button>
          </div>
          <p class="note">Key is stored locally in config.json and sent only to
          generativelanguage.googleapis.com. It is never echoed back to the page.</p>
        </form>
      </div>
    </div>
  </aside>
  <div class="backdrop" id="backdrop"></div>

  <main class="main">
    <div class="topbar">
      <button type="button" class="hamburger" id="hamburger" aria-label="Toggle sidebar">Menu</button>
      <span class="model-name">gemini-4-hydrogen</span>
      <button type="button" id="invert-top">Invert</button>
    </div>

    <div class="scroller" id="scroller" aria-live="polite">
      <div class="sysmsg" id="welcome">Session started. Prompts are proxied to
      gemini-4-hydrogen. Prefix a prompt with /sh to request a local shell
      command; it will ask for confirmation before running. Configure the API
      key under Settings.</div>
    </div>

    <div class="composer">
      <div class="comp-box">
        <span class="hint-chip" id="shchip">/sh shell command mode</span>
        <textarea id="prompt" rows="1" aria-label="Prompt"
          placeholder="Message gemini-4-hydrogen... (/sh for shell)"></textarea>
      </div>
      <div class="comp-row" style="margin-top:6px;">
        <span class="send"><button type="button" id="send" class="send" style="width:100%;">Send</button></span>
      </div>
      <p class="disclaimer">Designed by webxos.netlify.app. No Association with GOOGLE or GEMINI Software: Output is not verified by Google.</p>
    </div>
  </main>
</div>

<script>
(function () {
  "use strict";
  var scroller = document.getElementById("scroller");
  var promptEl = document.getElementById("prompt");
  var sendBtn = document.getElementById("send");
  var chip = document.getElementById("shchip");
  var pendingBusy = false;

  // ---- conversation list (in-memory, per 4.7: no unbounded storage) ----
  var convs = [];               // {title, preview}
  var convList = document.getElementById("convs");

  function refreshConvList() {
    convList.textContent = "";
    for (var i = 0; i < convs.length; i++) {
      var c = convs[i];
      var div = document.createElement("div");
      div.className = "conv" + (i === convs.length - 1 ? " active" : "");
      var t = document.createElement("div");
      t.className = "conv-title";
      t.textContent = c.title;
      var p = document.createElement("div");
      p.className = "conv-preview";
      p.textContent = c.preview;
      div.appendChild(t); div.appendChild(p);
      convList.appendChild(div);
    }
  }
  function addConv(title, preview) {
    convs.push({ title: title, preview: preview });
    if (convs.length > 40) { convs = convs.slice(-40); }  // bounded
    refreshConvList();
  }

  // ---- helpers ----
  function atBottom() {
    return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 40;
  }
  function scrollToBottom(force) {
    if (force || atBottom()) { scroller.scrollTop = scroller.scrollHeight; }
  }
  function addRow(kind) {
    var row = document.createElement("div");
    row.className = "msg-row " + kind;
    scroller.appendChild(row);
    return row;
  }
  function addUserMsg(text) {
    var row = addRow("user");
    var b = document.createElement("div");
    b.className = "bubble-user";
    b.textContent = text;
    row.appendChild(b);
    scrollToBottom(true);
    addConv(text.slice(0, 34), text.slice(0, 60));
  }
  function addSysMsg(text) {
    var row = addRow("ai");
    var d = document.createElement("div");
    d.className = "sysmsg";
    d.textContent = text;
    row.appendChild(d);
    scrollToBottom(false);
  }
  function addAiMsg(text, opts) {
    opts = opts || {};
    var row = addRow("ai");
    var b = document.createElement("div");
    b.className = "bubble-ai";
    if (opts.pre) {
      var label = document.createElement("div");
      label.textContent = opts.label || "Output";
      label.style.fontSize = "12px";
      label.style.color = "var(--muted)";
      b.appendChild(label);
      var pre = document.createElement("pre");
      pre.textContent = text;
      b.appendChild(pre);
    } else {
      b.textContent = text;
    }
    if (opts.token) {
      var actions = document.createElement("div");
      actions.className = "msg-actions";
      var run = document.createElement("button");
      run.type = "button"; run.textContent = "Run";
      var dis = document.createElement("button");
      dis.type = "button"; dis.textContent = "Discard";
      var note = document.createElement("span");
      note.className = "msg-meta";
      note.textContent = "This command is not on the read-only allowlist.";
      actions.appendChild(run); actions.appendChild(dis); actions.appendChild(note);
      b.appendChild(actions);
      run.addEventListener("click", function () { doExec(opts.token, b); });
      dis.addEventListener("click", function () {
        b.removeChild(actions);
        var m = document.createElement("div");
        m.className = "msg-meta"; m.textContent = "Command discarded.";
        b.appendChild(m);
      });
    }
    if (opts.copyable) {
      var cp = document.createElement("button");
      cp.type = "button"; cp.textContent = "Copy";
      cp.addEventListener("click", function () {
        // clipboard may be unavailable: fall back to selection copy
        try { navigator.clipboard.writeText(text); }
        catch (e) { document.execCommand("copy"); }
      });
      var meta = document.createElement("div");
      meta.className = "msg-actions";
      meta.appendChild(cp);
      b.appendChild(meta);
    }
    row.appendChild(b);
    scrollToBottom(false);
    return b;
  }
  var typingRow = null;
  function showTyping() {
    typingRow = addRow("ai");
    var d = document.createElement("div");
    d.className = "bubble-ai typing"; d.textContent = "typing...";
    typingRow.appendChild(d);
    scrollToBottom(true);
  }
  function hideTyping() {
    if (typingRow && typingRow.parentNode) { typingRow.parentNode.removeChild(typingRow); }
    typingRow = null;
  }

  // ---- API calls ----
  function post(path, obj) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(obj)
    }).then(function (r) {
      if (!r.ok) { throw new Error("HTTP " + r.status); }
      return r.json();
    });
  }

  function send() {
    var text = promptEl.value.replace(/\\s+$/, "");
    if (!text || pendingBusy) { return; }
    promptEl.value = "";
    autoResize();
    updateChip();
    addUserMsg(text);
    showTyping();
    pendingBusy = true;
    sendBtn.disabled = true;
    post("/api/run", { prompt: text }).then(function (data) {
      hideTyping();
      pendingBusy = false;
      sendBtn.disabled = false;
      if (data && data.confirmation) {
        addAiMsg(data.command || "", { token: data.token, label: "Proposed command", pre: true, copyable: true });
      } else if (data && typeof data.reply === "string") {
        addAiMsg(data.reply, { copyable: true });
      } else {
        addAiMsg("Empty reply.");
      }
    }).catch(function (err) {
      hideTyping();
      pendingBusy = false;
      sendBtn.disabled = false;
      addSysMsg("Request failed: " + err.message);
    });
  }

  function doExec(token, bubble) {
    post("/api/exec", { token: token }).then(function (data) {
      var out = document.createElement("pre");
      out.className = "cmd-out";
      out.textContent = data && data.output ? data.output
        : (data && data.error ? "Error: " + data.error : "(no output)");
      bubble.appendChild(out);
      scrollToBottom(false);
    }).catch(function (err) {
      addSysMsg("Execution request failed: " + err.message);
    });
  }

  // ---- composer behavior (4.3) ----
  function autoResize() {
    promptEl.style.height = "auto";
    var h = Math.min(promptEl.scrollHeight, 132);
    promptEl.style.height = Math.max(h, 44) + "px";
  }
  function updateChip() {
    var v = promptEl.value;
    var on = v === "/sh" || v.indexOf("/sh ") === 0;
    chip.classList.toggle("on", on);
  }
  promptEl.addEventListener("input", function () { autoResize(); updateChip(); });
  promptEl.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.altKey && !ev.ctrlKey) {
      ev.preventDefault();
      send();
    }
  });
  sendBtn.addEventListener("click", send);

  document.getElementById("newchat").addEventListener("click", function () {
    scroller.textContent = "";
    var w = document.createElement("div");
    w.className = "sysmsg";
    w.textContent = "New chat started.";
    scroller.appendChild(w);
    convs.forEach(function (c) { /* keep history list */ });
    addSysMsg("Prompts are proxied to gemini-4-hydrogen. /sh requests a local shell command.");
    closeSidebar();
  });

  // ---- 4.5 invert button (CSS variables, persisted) ----
  function applyThemeClass() {
    var root = document.documentElement;
    var dark = root.classList.contains("dark");
    document.getElementById("invert-side").textContent = dark ? "Light" : "Invert";
    document.getElementById("invert-top").textContent = dark ? "Light" : "Invert";
  }
  function invert() {
    var root = document.documentElement;
    root.classList.toggle("dark");
    var dark = root.classList.contains("dark");
    try { localStorage.setItem("harness-theme", dark ? "dark" : "light"); } catch (e) {}
    applyThemeClass();
  }
  document.getElementById("invert-top").addEventListener("click", invert);
  document.getElementById("invert-side").addEventListener("click", invert);
  applyThemeClass();

  // ---- settings drawer + config (4.6) ----
  var drawer = document.getElementById("drawer");
  document.getElementById("settings-btn").addEventListener("click", function () {
    drawer.classList.toggle("open");
    if (drawer.classList.contains("open")) {
      fetch("/api/config").then(function (r) { return r.json(); }).then(function (cfg) {
        var sel = document.getElementById("thinking");
        for (var i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === cfg.thinking) { sel.selectedIndex = i; break; }
        }
      }).catch(function () {});
    }
  });
  document.getElementById("cancel-cfg").addEventListener("click", function () {
    drawer.classList.remove("open");
  });
  document.getElementById("save-cfg").addEventListener("click", function () {
    var body = {
      apiKey: document.getElementById("apiKey").value,
      thinking: document.getElementById("thinking").value
    };
    post("/api/config", body).then(function (data) {
      document.getElementById("apiKey").value = "";
      drawer.classList.remove("open");
      addSysMsg("Configuration saved (thinking level: " + (data.thinking || "?") + ").");
    }).catch(function (err) {
      addSysMsg("Configuration not saved: " + err.message);
    });
  });

  // ---- OAuth capture (4.6) ----
  document.getElementById("auth-btn").addEventListener("click", function () {
    addSysMsg("Spawning Gemini CLI to capture the Google OAuth URL (up to 30 seconds)...");
    fetch("/api/cli-auth").then(function (r) { return r.json(); }).then(function (data) {
      if (data && data.url) {
        var row = addRow("ai");
        var d = document.createElement("div");
        d.className = "sysmsg";
        var a = document.createElement("a");
        a.href = data.url; a.target = "_blank"; a.rel = "noopener noreferrer";
        a.textContent = "Open Google login to complete OAuth";
        d.appendChild(a);
        row.appendChild(d);
      } else {
        addSysMsg(data && data.msg ? data.msg : "No OAuth URL captured.");
      }
    }).catch(function (err) {
      addSysMsg("OAuth capture failed: " + err.message);
    });
  });

  // ---- mobile sidebar drawer (4.1) ----
  var sidebar = document.getElementById("sidebar");
  var backdrop = document.getElementById("backdrop");
  function openSidebar() { sidebar.classList.add("open"); backdrop.classList.add("on"); }
  function closeSidebar() { sidebar.classList.remove("open"); backdrop.classList.remove("on"); }
  document.getElementById("hamburger").addEventListener("click", function () {
    if (sidebar.classList.contains("open")) { closeSidebar(); } else { openSidebar(); }
  });
  backdrop.addEventListener("click", closeSidebar);
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") { closeSidebar(); drawer.classList.remove("open"); }
  });

  // ---- initial config load (4.6) ----
  fetch("/api/config").then(function (r) { return r.json(); }).then(function (cfg) {
    var sel = document.getElementById("thinking");
    for (var i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === cfg.thinking) { sel.selectedIndex = i; break; }
    }
  }).catch(function () {});
  promptEl.focus();
})();
</script>
</body>
</html>"""


# --------------------------------------------------------------------------
# Gemini client (v1-M3: key in x-goog-api-key header, never in the URL;
# v1-H4: 60s timeout).
# --------------------------------------------------------------------------
def gemini_request(api_key, user_text, thinking, shell_mode):
    """Call the Gemini REST API; returns (reply_text, error_str)."""
    payload = {
        "systemInstruction": {"parts": [{"text": build_system_prompt(shell_mode)}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": thinking}},
    }
    req = urllib.request.Request(
        API_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,          # header auth, not query string
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=GEMINI_API_TIMEOUT) as r:
            res = json.loads(r.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read(4096).decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        detail = _redact(detail)
        return None, f"API Error {e.code}: {detail}"
    except urllib.error.URLError as e:
        return None, "API Error: network failure ({})".format(e.reason)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        return None, f"API Error: malformed response ({e})"
    except OSError as e:
        return None, f"API Error: {e}"

    candidates = res.get("candidates") or []
    if not candidates or not isinstance(candidates, list):
        return None, "API Error: no candidates in response."
    c0 = candidates[0] if isinstance(candidates[0], dict) else {}
    parts = (c0.get("content") or {}).get("parts") or []
    text = parts[-1].get("text") if parts and isinstance(parts[-1], dict) else None
    if not text:
        finish = c0.get("finishReason", "UNKNOWN")
        return None, f"API Error: no text in response (finishReason={finish})."
    return text, None


# --------------------------------------------------------------------------
# Execution (v1-L2: output truncated; v2-L3: allowlist runs logged).
# --------------------------------------------------------------------------
def run_shell(cmd):
    """Run a confirmed/allowlisted command; returns (output, error)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT,
        )
        out = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        out = out.strip()
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        return out, None
    except subprocess.TimeoutExpired:
        return None, f"command timed out after {EXEC_TIMEOUT}s"
    except OSError as e:
        return None, str(e)


# --------------------------------------------------------------------------
# CLI auth (v1-H5: 30s timeout + killpg; v1-L3: single-flight guard --
# only one npx spawn at a time, concurrent callers get a busy message).
# --------------------------------------------------------------------------
_CLI_AUTH_LOCK = threading.Lock()


def cli_auth_capture():
    if not _CLI_AUTH_LOCK.acquire(blocking=False):
        return {"url": None,
                "msg": "an OAuth capture is already in progress; try again after it finishes"}
    try:
        env = os.environ.copy()
        env["BROWSER"] = "echo"
        try:
            proc = subprocess.Popen(
                ["npx", "-y", "@google/gemini-cli"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
        except FileNotFoundError:
            return {"url": None,
                    "msg": "npx not found. Install Node.js or use the API key method in Settings."}
        except OSError as e:
            return {"url": None, "msg": f"could not launch npx: {e}"}

        try:
            out, _ = proc.communicate(timeout=CLI_AUTH_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
            return {"url": None,
                    "msg": f"CLI auth timed out after {CLI_AUTH_TIMEOUT}s. Try again or use the API key method."}
        except Exception as e:            # noqa: BLE001 -- last-resort guard
            return {"url": None, "msg": f"CLI auth failed: {e}"}

        url = next(
            (w for w in (out or "").split()
             if w.startswith("https://accounts.google.com")),
            None,
        )
        if url:
            LOG.info("captured Gemini CLI OAuth URL")
            return {"url": url, "msg": "Captured."}
        return {
            "url": None,
            "msg": "No OAuth URL found in CLI output. Install @google/gemini-cli or use the API key method in Settings.",
        }
    finally:
        _CLI_AUTH_LOCK.release()


# --------------------------------------------------------------------------
# HTTP Handler (v1-C1/H1 origin+host; v1-H2/v2-H2/v2-M1 body reads;
# v2-M5 CSP+nosniff on every response; v2-L2 close only when a body is
# actually pending).
# --------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HydrogenHarness/3.0"

    def log_message(self, fmt, *args):
        # v1-L1: per-request diagnostics to the rotating file log.
        try:
            LOG.info("%s %s", self.address_string(), _redact(fmt % args))
        except Exception:
            pass

    # ---------- request gating (v1-C1, v1-H1) ----------
    def _origin_ok(self):
        origin = self.headers.get("Origin")
        if origin is None:
            return True          # plain navigation / curl: no Origin header
        try:
            parsed = urllib.parse.urlparse(origin)
            host = (parsed.hostname or "").lower()
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            expected = [
                f"{HOST}:{self.server.server_address[1]}",
                f"localhost:{self.server.server_address[1]}",
            ]
            return f"{host}:{port}" in expected
        except ValueError:
            return False

    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0].strip().lower()
        return host in ("", HOST, "localhost", "[::1]", "::1")

    # ---------- responses (v2-M5: CSP + nosniff everywhere) ----------
    def _common_headers(self, content_type, length):
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP_POLICY)

    def _send(self, code, content_type, payload):
        self.send_response(code)
        self._common_headers(content_type, len(payload))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"))

    def _send_html(self, text, code=200):
        self._send(code, "text/html; charset=utf-8", text.encode("utf-8"))

    def _reject(self, code, reason):
        LOG.warning("rejected %s %s: %s", self.command, self.path, reason)
        # v2-L2: only close the connection when a body is really pending.
        try:
            raw_len = (self.headers.get("Content-Length") or "").strip()
            pending = int(raw_len) if raw_len.isdigit() else 0
        except ValueError:
            pending = 0
        if pending > 0:
            self.close_connection = True
        self._send_json({"error": reason}, code=code)

    # ---------- body reads (v1-H2, v2-H2, v2-M1) ----------
    def _read_body(self):
        """Read + parse the JSON body; capped; never silently loses input.

        Returns ({}, error) where error strings start with "413"/"400".
          ({}, None)       -- no body (fine)
          (None, "413...") -- body exceeds the cap; caller must reject
          (None, "400...") -- invalid Content-Length; caller must reject
          ({}, None)       -- body present but malformed JSON (treated as
                              empty: handlers see {} and respond safely)
        """
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return self._read_body_chunked()
        raw_len = (self.headers.get("Content-Length") or "").strip()
        if raw_len == "":
            return {}, None
        if not raw_len.isdigit():
            # v2-H2: garbage Content-Length is an error, not a silent 0.
            return None, "400 invalid Content-Length header"
        length = int(raw_len)
        if length == 0:
            return {}, None
        if length > MAX_BODY_BYTES:
            return None, f"413 request body too large (limit {MAX_BODY_BYTES} bytes)"
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8", errors="strict"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}, None
        return (body, None) if isinstance(body, dict) else ({}, None)

    def _read_body_chunked(self):
        """v2-M1: strict chunked framing (CRLF validated, capped)."""
        chunks = []
        total = 0
        try:
            while True:
                size_line = self.rfile.readline(1024)
                if not size_line:
                    return {}, None                 # peer went away
                try:
                    size = int(size_line.split(b";")[0].strip() or b"0", 16)
                except ValueError:
                    return None, "400 invalid chunk size"
                if size == 0:
                    while True:                     # trailers until blank line
                        line = self.rfile.readline(1024)
                        if not line or line in (b"\r\n", b"\n"):
                            break
                    break
                total += size
                if total > MAX_BODY_BYTES:
                    return None, f"413 request body too large (limit {MAX_BODY_BYTES} bytes)"
                data = self.rfile.read(size)
                if len(data) != size:
                    return None, "400 truncated chunk"
                crlf = self.rfile.readline(2)      # exactly the trailing CRLF
                if crlf not in (b"\r\n", b"\n"):
                    return None, "400 malformed chunk terminator"
                chunks.append(data)
            raw = b"".join(chunks)
            body = json.loads(raw.decode("utf-8", errors="strict"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}, None
        except OSError:
            return {}, None
        return (body, None) if isinstance(body, dict) else ({}, None)

    # ---------- GET ----------
    def do_GET(self):
        if not self._origin_ok() or not self._host_ok():
            self._reject(403, "forbidden: origin/host not allowed")
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send_html(HTML)
        elif path == "/api/config":
            cfg = load_cfg()
            key = cfg.get("apiKey", "")
            masked = ("*" * 8 + key[-4:]) if len(key) > 12 else ("set" if key else "")
            self._send_json({"apiKey": masked, "thinking": cfg.get("thinking", "high")})
        elif path == "/api/cli-auth":
            self._send_json(cli_auth_capture())
        else:
            self._reject(404, "not found")

    # ---------- POST ----------
    def do_POST(self):
        if not self._origin_ok() or not self._host_ok():
            self._reject(403, "forbidden: origin/host not allowed")
            return
        path = urllib.parse.urlparse(self.path).path
        body, body_err = self._read_body()
        if body_err:
            m = re.match(r"^(\d{3}) (.*)$", body_err, re.DOTALL)
            if m:
                code, msg = int(m.group(1)), m.group(2)
            else:
                code, msg = 400, body_err
            self._reject(code, msg)
            return

        if path == "/api/config":
            try:
                cfg = save_cfg(body)               # raises ValueError on bad input
                self._send_json({"ok": True, "thinking": cfg.get("thinking", "high")})
            except ValueError as e:                # v2-M2: 400, not silent drop
                LOG.warning("config rejected: %s", e)
                self._send_json({"error": f"invalid configuration: {e}"}, code=400)
            except OSError as e:
                LOG.warning("config save failed: %s", e)
                self._send_json({"error": f"could not save config: {e}"}, code=500)
            return

        if path == "/api/run":
            self.handle_run(body)
            return

        if path == "/api/exec":
            self.handle_exec(body)
            return

        self._reject(404, "not found")

    # ---------- /api/run ----------
    def handle_run(self, body):
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json({"reply": "Error: empty prompt."})
            return
        prompt = prompt.strip()
        cfg = load_cfg()
        api_key = cfg.get("apiKey", "")
        if not api_key:
            self._send_json({"reply": "Error: no API key configured. Open Settings in the sidebar and save one, or set GEMINI_API_KEY."})
            return

        shell_mode, user_text = split_prompt(prompt)     # v1-M1
        if shell_mode and not user_text:
            self._send_json({"reply": "Error: /sh requires a request, e.g. '/sh show disk usage'."})
            return

        reply, api_err = gemini_request(
            api_key, user_text, cfg.get("thinking", "high"), shell_mode
        )
        if api_err:
            LOG.warning("gemini call failed: %s", api_err)
            self._send_json({"reply": api_err})
            return

        if not shell_mode:
            self._send_json({"reply": reply})
            return

        # ---- shell mode: validate before ANY execution ----
        if is_refusal(reply):                            # v1-M2
            LOG.info("refusal detected; not executed")
            self._send_json({"reply": "The model declined to produce a command. Its reply:\n" + reply})
            return

        cmd = extract_command(reply)                     # v1-M1
        if not cmd or not looks_like_command(cmd):       # v1-M2
            self._send_json({"reply": "The model's reply was not a recognizable shell command. Its reply:\n" + reply})
            return

        risk = command_risk(cmd)                        # v2-C1 + v2-L1 inside
        if risk == "blocked":
            LOG.warning("blocked dangerous command: %r", cmd)
            self._send_json({"reply": "Refused to run (matches the harness deny list):\n" + cmd})
            return

        if risk == "safe":
            # read-only allowlist: single line, no metacharacters
            out, exec_err = run_shell(cmd)
            if exec_err:
                self._send_json({"reply": f"Execution Error: {exec_err}\nCommand: {cmd}"})
            else:
                LOG.info("auto-executed safe command: %r", cmd)   # v2-L3
                self._send_json({"reply": f"Command executed (allowlisted, read-only):\n{cmd}\n--- OUTPUT ---\n{out}"})
            return

        # risk == "confirm": one-time token gate (v1-C2)
        token = _register_pending(cmd, user_text)
        LOG.info("awaiting confirmation for command: %r", cmd)
        self._send_json({
            "confirmation": True,
            "command": cmd,
            "token": token,
            "reply": "The model proposed a command that needs your confirmation.",
        })

    # ---------- /api/exec ----------
    def handle_exec(self, body):
        token = body.get("token")
        cmd = _pop_pending(token)
        if cmd is None:
            self._send_json({"error": "invalid or expired confirmation token"}, code=200)
            return
        risk = command_risk(cmd)
        if risk == "blocked":
            LOG.warning("confirmed command hit deny list: %r", cmd)
            self._send_json({"error": "refused: command matches the harness deny list"}, code=200)
            return
        out, exec_err = run_shell(cmd)
        if exec_err:
            self._send_json({"error": f"Execution Error: {exec_err}"}, code=200)
            return
        LOG.info("executed confirmed command: %r", cmd)
        self._send_json({"output": out or "(no output)"})


# --------------------------------------------------------------------------
# Server bootstrap (n9): 127.0.0.1 only, port from env/CLI, fallback chain.
# --------------------------------------------------------------------------
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _port_from_env():
    raw = (os.environ.get("HARNESS_PORT") or "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        print(f"Warning: HARNESS_PORT={raw!r} is not an integer; ignoring.", file=sys.stderr)
        return None
    if not (1 <= port <= 65535):
        print(f"Warning: HARNESS_PORT={raw!r} out of range; ignoring.", file=sys.stderr)
        return None
    return port


def _bind_server(port):
    """Try to bind HOST:port; on OSError fall through the port chain (n9)."""
    candidates = [port] + [p for p in FALLBACK_PORTS if p != port]
    last_err = None
    for p in candidates:
        try:
            return ThreadedTCPServer((HOST, p), Handler), p
        except OSError as e:
            last_err = e
            continue
    return None, last_err


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="harness.py",
        description="Gemini 4 Hydrogen // Minimal Harness (serves 127.0.0.1 only).",
    )
    parser.add_argument(
        "--port", type=int, default=None, metavar="N",
        help=f"port to listen on (default {DEFAULT_PORT}; env HARNESS_PORT; "
             f"falls back to {FALLBACK_PORTS} if busy)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="do not open a browser window automatically",
    )
    args = parser.parse_args(argv)

    port = args.port if args.port is not None else _port_from_env()
    if port is None:
        port = DEFAULT_PORT
    if not (1 <= port <= 65535):
        print(f"Error: invalid port {port}.", file=sys.stderr)
        return 2

    httpd, bound = _bind_server(port)
    if httpd is None:
        print(f"Error: could not bind {HOST} on any port in {FALLBACK_PORTS}: {bound}", file=sys.stderr)
        return 1
    if bound != port:
        print(f"Notice: port {port} unavailable; using {bound} instead.")

    print(f"Gemini 4 Hydrogen Harness running at http://{HOST}:{bound}")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Log file: {LOG_FILE}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{bound}")).start()

    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
