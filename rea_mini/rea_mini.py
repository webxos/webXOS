#!/usr/bin/env python3
"""
REA_mini.py — Reverse-engineer-anything mini harness (single-file megalith).

Inspired by morluto/rea. Free tools + free public APIs first.

Menu
----
  1. Capture URL -> clean Markdown (LLM-friendly)
  2. Capture URL -> self-contained HTML archive
  3. Search fan-out across free public APIs
  4. Ghidra headless analysis of a local binary + keyword search
  5. Generate skeleton file (HTML/JS/Python/Rust/C++) from idea/repo/URL
  6. Full pipeline: URL -> fan-out -> synthesize -> skeleton
  7. Self-check: report every missing dependency with exact fix command
  8. Quit

Non-interactive / programmatic modes
------------------------------------
  --cmd capture|archive|fanout|ghidra|generate|full|selfcheck|mcp|validate-evidence
  --cmd mcp : JSON-RPC 2.0 MCP server over stdio (tools for agents)
  --list-models : show resolved LLM configuration (local + optional cloud)
  --cmd validate-evidence --evidence-file <path> : check evidence sidecar schema

Onboarding / local-first LLM
----------------------------
* On interactive startup, the harness REQUIRES a local Ollama model
  (default `qwen2.5:0.5b`). It auto-detects the Ollama daemon, starts it
  if necessary, and pulls the model if missing. Choice is persisted in
  `rea_mini_config.json` and `ollama_cache.json`.
* After the local model works, it asks ONE optional question:
  "EXPAND LLM? Would you like to also add a cloud model?" Cloud providers
  are strictly opt-in and layered behind the local fallback. Declining is
  remembered; re-ask only if `--list-models` explicitly enables them.
* Every dependency check re-uses the self-check framework, then offers a
  single "[A]ll / [N]one / [I]nteractive" install prompt so users don't
  answer ten questions.
* `--no-require-local-llm` opts out of the local requirement (falls back
  to cloud or deterministic templates) — useful for CI.
* Non-interactive `--cmd` invocations never run the wizard. LLM-requiring
  commands fail fast with EXIT_NO_LOCAL_LLM (10) unless a cloud provider
  or `--no-require-local-llm` is present.

Useful flags
------------
  -y/--yes/--auto-install : install missing deps & pull model without prompting
  -q/--quiet              : suppress progress banners (results still printed)
  --scroll                : scroll to bottom before capturing (capture command)
  --wait-selector <css>   : wait for CSS selector before capturing
  --run-generated         : also run generated code's --test mode (executes it)
  --no-require-local-llm  : do not require a local Ollama model

Design notes
------------
* Every artifact-producing command writes a `<artifact>.evidence.json` sidecar
  in the schema {schema_version, claim, evidence, limitations, unknowns,
  reproducibility, generated_at}.
* Ghidra runs write a small Java postScript (`ghidra_scripts/ReaDump.java`)
  that dumps functions + printable strings to JSON; the `-analysisTimeoutPerFile`
  flag is only passed when analyzeHeadless supports it, and that support is
  cached in `ghidra_cache.json`.
* Skeleton generation asks the LLM for a working core and falls back to
  deterministic, runnable templates. Every artifact is syntax-checked
  (safe, no execution) and — only when `--run-generated` is passed — also
  smoke-tested by running its own `--test` mode.
* Fan-out is cached (1h TTL), ranked by keyword overlap with per-source
  weights, then MMR-diversified (O(cap^3) in the head; see inline comment).
* Evidence misreports are prevented by tracking `used_playwright` explicitly.
* MCP stdio server exposes the same tools for agents; stdin reads run off
  the asyncio loop so slow tool calls do not wedge the server.
* LLM model names are configurable via `REA_LLM_*` env vars and config.

Exit codes: 0 ok, 2 py<3.9, 3 missing deps, 4 bad url/input,
            5 nav fail, 6 extract empty, 7 write fail, 8 archive fail,
            9 bad evidence file, 10 local LLM required but unavailable.
"""

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shutil
import site
import string
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from html import escape as _html_escape
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional third-party imports — the script always loads.
# ---------------------------------------------------------------------------
class _PWTimeoutStub(Exception):
    pass

class _PWErrorStub(Exception):
    pass

PLAYWRIGHT_OK = False
READABILITY_OK = False
HTML2TEXT_OK = False
async_playwright = None
PWTimeout = _PWTimeoutStub
PWError = _PWErrorStub
_Doc = None
_h2t = None


def _detect_optional_deps():
    """(Re-)detect optional deps in-process; safe to call repeatedly."""
    global PLAYWRIGHT_OK, async_playwright, PWTimeout, PWError
    global READABILITY_OK, _Doc
    global HTML2TEXT_OK, _h2t
    try:
        from playwright.async_api import (
            async_playwright as _ap,
            TimeoutError as _pt,
            Error as _pe,
        )
        async_playwright = _ap
        PWTimeout = _pt
        PWError = _pe
        PLAYWRIGHT_OK = True
    except ImportError:
        async_playwright = None
        PWTimeout = _PWTimeoutStub
        PWError = _PWErrorStub
        PLAYWRIGHT_OK = False
    try:
        from readability import Document as _d
        _Doc = _d
        READABILITY_OK = True
    except ImportError:
        _Doc = None
        READABILITY_OK = False
    try:
        import html2text as _hh
        _h2t = _hh
        HTML2TEXT_OK = True
    except ImportError:
        _h2t = None
        HTML2TEXT_OK = False


_detect_optional_deps()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "mode": "llm",
    "timeout": 30,
    "output_dir": "rea_output",
    "ignore_images": False,
    "log_max_bytes": 5 * 1024 * 1024,
    "user_agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "http_timeout": 20,
    "fanout_cache_ttl_s": 3600,
    "archive_cli_version": "2.0.75",
    "scroll_to_bottom": False,
    "wait_selector": "",
    "mmr_lambda": 0.7,
    "mmr_cap": 100,
    "archive_timeout_s": 600,
    # Local-first LLM (onboarding wizard persists these)
    "require_local_llm": True,
    "llm_fallback_model": "",
    "cloud_provider": "",
    "cloud_model": "",
    "ollama_probe_timeout_s": 2,
    "strict_local_llm_skip_cloud": True,
}

EXIT_OK, EXIT_PYTHON_OLD, EXIT_MISSING_DEPS = 0, 2, 3
EXIT_BAD_URL, EXIT_NAV_FAIL, EXIT_EXTRACT_EMPTY = 4, 5, 6
EXIT_WRITE_FAIL, EXIT_ARCHIVE_FAIL = 7, 8
EXIT_BAD_EVIDENCE = 9
EXIT_NO_LOCAL_LLM = 10

GHIDRA_RELEASE_API = "https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/latest"
GHIDRA_CACHE_FILE = "ghidra_cache.json"

# Local-first LLM constants
STARTER_MODEL = "qwen2.5:0.5b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CACHE_FILE = "ollama_cache.json"

# Global quiet flag — set once in async_main, read by _say().
_QUIET = False


def _say(*args, **kwargs):
    """Progress banner — suppressed under --quiet. Results still use print()."""
    if not _QUIET:
        print(*args, **kwargs)


# Configurable LLM models. Environment variables win over config.
DEFAULT_LLM_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "deepseek": "deepseek-chat",
    "mistral": "mistral-small-latest",
    "gemini": "gemini-2.0-flash",
    "huggingface": "HuggingFaceH4/zephyr-7b-beta",
    "ollama": STARTER_MODEL,
}

LLM_MODEL_ENV = {
    "groq": "REA_LLM_GROQ_MODEL",
    "openrouter": "REA_LLM_OPENROUTER_MODEL",
    "openai": "REA_LLM_OPENAI_MODEL",
    "anthropic": "REA_LLM_ANTHROPIC_MODEL",
    "deepseek": "REA_LLM_DEEPSEEK_MODEL",
    "mistral": "REA_LLM_MISTRAL_MODEL",
    "gemini": "REA_LLM_GEMINI_MODEL",
    "huggingface": "REA_LLM_HF_MODEL",
    "ollama": "REA_LLM_OLLAMA_MODEL",
}

# Env var whose presence means "this provider is configured". None = no key.
PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "huggingface": "HF_TOKEN",
    "ollama": None,
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Log:
    def __init__(self, path: Path, max_bytes: int):
        self.path = path
        self.max_bytes = max_bytes
        self.fh = None
        self._rotate()
        try:
            self.fh = open(self.path, "a", encoding="utf-8", buffering=1)
        except OSError as e:
            print(f"Warning: cannot open log {self.path}: {e}", file=sys.stderr)

    def _rotate(self):
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                os.replace(self.path, self.path.with_name(self.path.name + ".1"))
        except OSError as e:
            # Surface one warning; log rotation failures shouldn't abort the run.
            print(f"Warning: log rotation failed for {self.path}: {e}", file=sys.stderr)

    def _line(self, level, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{ts} {level:<5} {msg}\n"
        if self.fh and not self.fh.closed:
            try:
                self.fh.write(line)
            except OSError:
                pass
        return line

    def info(self, m):
        return self._line("INFO", m)

    def warn(self, m):
        return self._line("WARN", m)

    def error(self, m):
        return self._line("ERROR", m)

    def fatal(self, m):
        return self._line("FATAL", m)

    def block(self, level, lines):
        if not lines:
            return
        self._line(level, lines[0])
        if self.fh and not self.fh.closed:
            for x in lines[1:]:
                try:
                    self.fh.write("    " + x + "\n")
                except OSError:
                    pass

    def close(self):
        if self.fh and not self.fh.closed:
            try:
                self.fh.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_config(path: Path, log):
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                u = json.load(f)
            if isinstance(u, dict):
                cfg.update(u)
            log.info(f"config_loaded path={path}")
        except (json.JSONDecodeError, OSError) as e:
            log.warn(f"config_invalid err={type(e).__name__}")
    else:
        log.info(f"config_missing path={path}")
    return cfg


def save_config(path: Path, cfg: dict, log=None):
    """Persist config atomically. Failures are logged but never fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        if log:
            log.info(f"config_saved path={path}")
    except OSError as e:
        if log:
            log.warn(f"config_save_failed path={path} err={type(e).__name__}")


def normalize_url(raw):
    raw = (raw or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "https://" + raw
    try:
        p = urllib.parse.urlparse(raw)
    except ValueError:
        return None
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    return raw


def redact_url(u):
    try:
        p = urllib.parse.urlparse(u)
        return p._replace(query="<redacted>").geturl() if p.query else u
    except Exception:
        return "<unparseable>"


def slugify(url, mode, ext=None):
    p = urllib.parse.urlparse(url)
    d = re.sub(r"[^A-Za-z0-9]+", "_", p.netloc or "site")
    s = re.sub(r"[^A-Za-z0-9]+", "_", p.path.strip("/") or "index")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    e = ext or ("md" if mode == "llm" else "html")
    stem = f"{d}_{s}_{mode}_{ts}"
    max_stem = max(1, 200 - len(e) - 1)
    if len(stem) > max_stem:
        stem = stem[:max_stem]
    return f"{stem}.{e}"


def atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _cfg_int(cfg, key, default, log=None):
    try:
        return int(cfg.get(key, default))
    except (TypeError, ValueError):
        if log:
            log.warn(f"config_int_invalid key={key} value={cfg.get(key)!r} using_default={default}")
        return default


def _cfg_float(cfg, key, default, log=None):
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        if log:
            log.warn(f"config_float_invalid key={key} value={cfg.get(key)!r} using_default={default}")
        return default


def _cfg_str(cfg, key, default):
    v = cfg.get(key, default)
    return v if isinstance(v, str) else default


def _github_headers(accept="application/vnd.github+json"):
    h = {"User-Agent": "REA_mini.py/1.1", "Accept": accept}
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h


def http_get_json(url, timeout=20, headers=None):
    req = urllib.request.Request(url, headers=headers or {
        "User-Agent": "REA_mini.py/1.1",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _safe_get_json(url, timeout=20, headers=None):
    try:
        return http_get_json(url, timeout=timeout, headers=headers)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _refresh_sys_path():
    """Make user-site packages importable in-process."""
    try:
        paths = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        return
    for p in paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _try_reimport_optional():
    """Invalidate cached optional modules and re-run detection."""
    for name in list(sys.modules.keys()):
        if name == "playwright" or name.startswith("playwright.") \
           or name == "readability" or name == "html2text":
            try:
                del sys.modules[name]
            except KeyError:
                pass
    _refresh_sys_path()
    _detect_optional_deps()


def keywords_from(text, limit=8):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text or "")
    stop = {"the", "and", "for", "with", "from", "that", "this", "into",
            "how", "what", "why", "when", "where", "which", "make", "made",
            "using", "have", "will", "would", "could", "should", "about",
            "your", "their", "there", "these", "those", "then", "than",
            "http", "https", "www", "com", "org", "net", "html", "htm"}
    seen, out = set(), []
    for t in tokens:
        tl = t.lower()
        if tl in stop or tl in seen:
            continue
        seen.add(tl)
        out.append(t)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Evidence envelope
# ---------------------------------------------------------------------------
def _reproducibility_block(argv=None):
    return {
        "argv": list(argv) if argv is not None else list(sys.argv),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": f"{platform.system()} {platform.machine()}",
        "cwd": str(Path.cwd()),
        "optional_deps": {
            "playwright": PLAYWRIGHT_OK,
            "readability_lxml": READABILITY_OK,
            "html2text": HTML2TEXT_OK,
        },
    }


def evidence_wrap(claim, evidence, limitations=None, unknowns=None,
                  extra=None, reproducibility=None):
    return {
        "schema_version": "1.1",
        "claim": claim,
        "evidence": evidence if evidence is not None else [],
        "limitations": limitations or [],
        "unknowns": unknowns or [],
        "extra": extra or {},
        "reproducibility": reproducibility or _reproducibility_block(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _evidence_sidecar_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + ".evidence.json")


def _write_evidence_sidecar(artifact: Path, evidence, log=None) -> Path:
    path = _evidence_sidecar_path(artifact)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False, default=str)
        return path
    except OSError as e:
        if log:
            log.warn(f"evidence_sidecar_write_failed artifact={artifact} err={e}")
        return path


EVIDENCE_REQUIRED_KEYS = ("schema_version", "claim", "evidence",
                          "limitations", "unknowns", "generated_at")
EVIDENCE_LIST_KEYS = ("evidence", "limitations", "unknowns")


def cmd_validate_evidence(path, log):
    p = Path(path).expanduser()
    if not p.is_file():
        print(f"Not a file: {p}")
        return EXIT_BAD_EVIDENCE
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to parse {p}: {e}")
        return EXIT_BAD_EVIDENCE
    if not isinstance(data, dict):
        print(f"INVALID: {p} root is not a JSON object")
        return EXIT_BAD_EVIDENCE
    missing = [k for k in EVIDENCE_REQUIRED_KEYS if k not in data]
    if missing:
        print(f"INVALID: missing required keys: {missing}")
        return EXIT_BAD_EVIDENCE
    for k in EVIDENCE_LIST_KEYS:
        if not isinstance(data[k], list):
            print(f"INVALID: '{k}' is not a list")
            return EXIT_BAD_EVIDENCE
    print(f"VALID: schema_version={data['schema_version']} "
          f"evidence_items={len(data['evidence'])} "
          f"limitations={len(data['limitations'])} "
          f"unknowns={len(data['unknowns'])}")
    if log:
        log.info(f"evidence_validated path={p}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Probes & installers
# ---------------------------------------------------------------------------
async def probe_chromium(timeout=10):
    if not PLAYWRIGHT_OK:
        return False, "playwright_not_installed"

    async def _inner():
        async with async_playwright() as p:
            path = p.chromium.executable_path
            if path and os.path.exists(path):
                return True, path
            return False, f"chromium_binary_missing expected={path}"

    try:
        return await asyncio.wait_for(_inner(), timeout=timeout)
    except asyncio.TimeoutError:
        return False, "chromium_probe_timeout"
    except Exception as e:
        return False, f"playwright_probe_failed err={type(e).__name__}"


def probe_npx():
    return bool(shutil.which("npx") and shutil.which("node"))


def probe_strings():
    return bool(shutil.which("strings"))


def probe_java():
    if not shutil.which("java"):
        return False, "java_not_found"
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
    except Exception as e:
        return False, f"java_probe_failed {type(e).__name__}"
    out = (r.stderr or "") + (r.stdout or "")
    if r.returncode != 0:
        return False, f"java_version_failed rc={r.returncode}"
    m = re.search(r'version "(\d+)', out)
    if not m:
        return False, "java_version_unparsed"
    major = int(m.group(1))
    return (major >= 21), f"java_{major}"


def _analyze_headless_path(ghidra_path: Path):
    for name in ("analyzeHeadless", "analyzeHeadless.bat"):
        p = ghidra_path / "support" / name
        if p.exists():
            return p
    return None


def check_ghidra_timeout_flag(ghidra_path: Path) -> bool:
    """Probe once for -analysisTimeoutPerFile support. Callers should cache."""
    hl = _analyze_headless_path(ghidra_path)
    if not hl:
        return False
    try:
        r = subprocess.run([str(hl), "-help"], capture_output=True, text=True, timeout=30)
        combined = (r.stdout or "") + (r.stderr or "")
        return "-analysisTimeoutPerFile" in combined
    except Exception:
        return False


# --- Ghidra cache ---------------------------------------------------------
def _ghidra_cache_path(script_dir: Path) -> Path:
    return script_dir / GHIDRA_CACHE_FILE


def _load_ghidra_cache(script_dir: Path) -> dict:
    p = _ghidra_cache_path(script_dir)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ghidra_cache(script_dir: Path, cache: dict):
    p = _ghidra_cache_path(script_dir)
    try:
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, p)
    except OSError:
        pass


def find_ghidra(script_dir: Path):
    candidates = []
    if os.environ.get("GHIDRA_HOME"):
        candidates.append(Path(os.environ["GHIDRA_HOME"]))

    # Cached path first — tolerate corrupted cache entries.
    cache = _load_ghidra_cache(script_dir)
    cached = cache.get("path")
    if cached:
        try:
            candidates.append(Path(cached))
        except (TypeError, ValueError):
            pass

    for base in (script_dir, Path.home(), Path("/opt"), Path("/usr/local")):
        candidates.append(base / "ghidra")
        try:
            for p in base.glob("ghidra_*"):
                if p.is_dir():
                    candidates.append(p)
        except OSError:
            pass
    dl_dir = script_dir / "ghidra_dl"
    if dl_dir.exists():
        try:
            for p in dl_dir.glob("ghidra_*"):
                if p.is_dir():
                    candidates.append(p)
        except OSError:
            pass

    seen = set()
    for c in candidates:
        try:
            rp = c.resolve()
        except OSError:
            rp = c
        if rp in seen:
            continue
        seen.add(rp)
        if c.is_dir() and _analyze_headless_path(c):
            return c
    return None


def download_ghidra(dest_dir: Path, log: Log, script_dir: Path = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"ghidra_download_start dest={dest_dir}")
    try:
        meta = http_get_json(GHIDRA_RELEASE_API, headers=_github_headers())
    except Exception as e:
        log.error(f"ghidra_download_failed stage=meta err={e}")
        raise

    zip_url = None
    expected_digest = None
    release_tag = meta.get("tag_name") if isinstance(meta, dict) else None
    for asset in meta.get("assets", []):
        n = asset.get("name", "")
        if n.endswith(".zip") and "ghidra" in n.lower() and "PUBLIC" in n.upper():
            zip_url = asset.get("browser_download_url")
            expected_digest = asset.get("digest")
            break
    if not zip_url:
        for asset in meta.get("assets", []):
            if asset.get("name", "").endswith(".zip"):
                zip_url = asset.get("browser_download_url")
                expected_digest = asset.get("digest")
                break
    if not zip_url:
        log.error("ghidra_download_failed reason=no_zip_asset")
        raise RuntimeError("no Ghidra zip asset in latest release")

    zip_path = dest_dir / "ghidra_release.zip"
    _say(f"Downloading Ghidra (~400 MB) from {zip_url} ...")
    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "REA_mini.py/1.1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(zip_path, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            last = 0
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done - last >= 1024 * 1024:
                    last = done
                    mb = done // (1024 * 1024)
                    if total:
                        pct = 100.0 * done / total
                        _say(f"\r  {mb} / {total // (1024*1024)} MB ({pct:.1f}%)",
                             end="", flush=True)
                    else:
                        _say(f"\r  {mb} MB", end="", flush=True)
        _say()
    except Exception as e:
        log.error(f"ghidra_download_failed stage=fetch err={e}")
        raise

    digest = None
    try:
        digest = _sha256_file(zip_path)
        log.info(f"ghidra_download_sha256 file={zip_path.name} sha256={digest}")
        print(f"SHA256: {digest}")
        if expected_digest and expected_digest.startswith("sha256:"):
            want = expected_digest.split(":", 1)[1].lower()
            if want != digest.lower():
                log.error(f"ghidra_download_integrity_failed expected={want} got={digest}")
                raise RuntimeError("Ghidra zip SHA256 mismatch (published digest)")
            log.info("ghidra_download_integrity_ok")
    except OSError as e:
        log.warn(f"ghidra_download_hash_failed err={e}")

    _say(f"Extracting {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path) as z:
            dest_resolved = dest_dir.resolve()
            for name in z.namelist():
                try:
                    target = (dest_dir / name).resolve()
                    target.relative_to(dest_resolved)
                except (ValueError, OSError):
                    log.error(f"ghidra_extract_skipped reason=zipslip name={name}")
                    continue
                z.extract(name, dest_dir)
    except Exception as e:
        log.error(f"ghidra_extract_failed err={e}")
        raise
    zip_path.unlink(missing_ok=True)

    for p in dest_dir.iterdir():
        if p.is_dir() and p.name.startswith("ghidra_") and _analyze_headless_path(p):
            log.info(f"ghidra_ready path={p}")
            if script_dir is not None:
                cache = _load_ghidra_cache(script_dir)
                cache.update({
                    "path": str(p),
                    "sha256": digest,
                    "release_tag": release_tag,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                })
                _save_ghidra_cache(script_dir, cache)
            return p
    log.error("ghidra_download_failed reason=extracted_dir_not_found")
    raise RuntimeError("Ghidra extracted but analyzeHeadless not found")


def auto_install_pip(pkgs, log: Log):
    if not pkgs:
        return True
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
    if not in_venv:
        cmd.append("--user")
    cmd.extend(pkgs)
    print(f"Installing pip packages: {' '.join(pkgs)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            log.info(f"pip_install_ok pkgs={pkgs}")
            return True
        log.error(f"pip_install_failed pkgs={pkgs} stderr={(r.stderr or '')[-500:]}")
        return False
    except Exception as e:
        log.error(f"pip_install_exception pkgs={pkgs} err={e}")
        return False


# ---------------------------------------------------------------------------
# Local LLM (Ollama) bootstrap — persisted in ollama_cache.json
# ---------------------------------------------------------------------------
def _ollama_cache_path(script_dir: Path) -> Path:
    return script_dir / OLLAMA_CACHE_FILE


def _load_ollama_cache(script_dir: Path) -> dict:
    p = _ollama_cache_path(script_dir)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ollama_cache(script_dir: Path, updates: dict):
    if script_dir is None:
        return
    cache = _load_ollama_cache(script_dir)
    cache.update(updates)
    cache["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        p = _ollama_cache_path(script_dir)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, p)
    except OSError:
        pass


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def _ollama_health(host=None, timeout=2.0):
    """Return (healthy, host_or_reason)."""
    host = (host or _ollama_host()).rstrip("/")
    try:
        data = _safe_get_json(f"{host}/api/tags", timeout=timeout)
        return (data is not None), host
    except Exception as e:
        return False, f"probe_err={type(e).__name__}"


def _ollama_list_models(host=None, timeout=2):
    host = (host or _ollama_host()).rstrip("/")
    try:
        data = http_get_json(f"{host}/api/tags", timeout=timeout)
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _ollama_pull(model, host=None, log=None):
    """Stream /api/pull until completion. Returns (ok, reason)."""
    host = (host or _ollama_host()).rstrip("/")
    body = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/pull", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    last_status = ""
    last_pct = -1
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                try:
                    ev = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if ev.get("error"):
                    if log:
                        log.error(f"ollama_pull_error model={model} err={ev['error']}")
                    return False, f"error:{ev['error']}"
                st = ev.get("status", "")
                total = ev.get("total") or 0
                completed = ev.get("completed") or 0
                if total and completed:
                    pct = int(100 * completed / total)
                    if pct != last_pct:
                        _say(f"\r  {st} {pct}% "
                             f"({completed // (1024*1024)} / {total // (1024*1024)} MB)",
                             end="", flush=True)
                        last_pct = pct
                elif st and st != last_status:
                    _say(f"\r  {st}", end="", flush=True)
                last_status = st
        _say()
        if log:
            log.info(f"ollama_pull_ok model={model}")
        return True, "pulled"
    except Exception as e:
        _say()
        if log:
            log.error(f"ollama_pull_failed model={model} err={type(e).__name__}")
        return False, f"{type(e).__name__}"


def _start_ollama_server():
    """Best-effort attempt to launch the local Ollama daemon."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", "-a", "Ollama"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Windows":
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=0x00000008)  # DETACHED_PROCESS
        else:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        return True
    except Exception:
        return False


async def _ensure_local_llm(script_dir, config, log, interactive=True, auto_pull=False):
    """Ensure a working local Ollama model is available.

    Returns (ok, model_name_or_reason_string).
    """
    if not config.get("require_local_llm", True):
        return True, "not_required"

    configured_model = (config.get("llm_fallback_model") or "").strip() or STARTER_MODEL
    host = _ollama_host()

    _say("REA_mini needs a local Ollama model to run. Checking…")

    if not shutil.which("ollama"):
        _say("Ollama is not installed. Install it from https://ollama.com/download, then re-run.")
        if log:
            log.error("local_llm_missing reason=ollama_binary_not_found")
        return False, "ollama_binary_missing"

    ok, info = _ollama_health(
        host=host,
        timeout=_cfg_int(config, "ollama_probe_timeout_s", 2, log))
    if not ok:
        _say("Ollama server is not reachable. Attempting to start it…")
        if not _start_ollama_server():
            _say("Could not start Ollama automatically. Run `ollama serve` and re-run.")
            if log:
                log.error("ollama_start_failed")
            return False, "ollama_server_down"
        for _ in range(6):
            await asyncio.sleep(1.5)
            ok, info = _ollama_health(host=host, timeout=2)
            if ok:
                break
        if not ok:
            _say("Ollama server did not come up. Run `ollama serve` and re-run.")
            if log:
                log.error("ollama_server_down_after_start")
            return False, "ollama_server_down"

    models = _ollama_list_models(host=host)
    chosen = configured_model

    if chosen not in models:
        if not interactive and not auto_pull:
            if log:
                log.error(f"local_llm_model_missing model={chosen}")
            return False, f"model_missing:{chosen}"
        if interactive and not auto_pull:
            _say(f"Recommended starter model ({chosen}) is not installed.")
            ans = _ask(f"Pull {chosen} now? [Y/n]: ", "y").lower()
            if ans not in ("y", "yes", ""):
                _say("Cannot proceed without a local model.")
                return False, "user_declined_pull"
        _say(f"Pulling {chosen} (this may take a few minutes)…")
        ok, reason = await asyncio.to_thread(_ollama_pull, chosen, host, log)
        if not ok:
            _say(f"Pull failed: {reason}. Run `ollama pull {chosen}` manually and re-run.")
            return False, f"pull_failed:{reason}"
        models = _ollama_list_models(host=host)

    if chosen not in models:
        return False, f"model_still_missing:{chosen}"

    _say(f"Verifying local model {chosen}…")
    test = await asyncio.to_thread(
        _llm_ollama, "Reply with exactly the word ok.", chosen, log)
    if not test or not test.strip():
        if log:
            log.error(f"local_llm_smoke_failed model={chosen}")
        return False, "model_smoke_failed"

    config["llm_fallback_model"] = chosen
    _save_ollama_cache(script_dir, {"model": chosen, "host": host})
    _say(f"Local model ready: {chosen}")
    if log:
        log.info(f"local_llm_ready model={chosen} host={host}")
    return True, chosen


async def _cloud_opt_in(script_dir, config, log):
    """Optional 'EXPAND LLM' step. Remembered across runs via ollama_cache.json."""
    cache = _load_ollama_cache(script_dir)
    if cache.get("cloud_prompted") or config.get("cloud_provider"):
        return

    ans = _ask("EXPAND LLM? Would you like to also add a cloud model? [y/N]: ", "n").lower()
    if ans not in ("y", "yes"):
        _say("Staying local-only. You can enable a cloud model later with --list-models.")
        _save_ollama_cache(script_dir, {"cloud_prompted": True})
        return

    providers = [
        ("groq", "GROQ_API_KEY", DEFAULT_LLM_MODELS["groq"]),
        ("openrouter", "OPENROUTER_API_KEY", DEFAULT_LLM_MODELS["openrouter"]),
        ("openai", "OPENAI_API_KEY", DEFAULT_LLM_MODELS["openai"]),
        ("anthropic", "ANTHROPIC_API_KEY", DEFAULT_LLM_MODELS["anthropic"]),
        ("deepseek", "DEEPSEEK_API_KEY", DEFAULT_LLM_MODELS["deepseek"]),
        ("mistral", "MISTRAL_API_KEY", DEFAULT_LLM_MODELS["mistral"]),
        ("gemini", "GEMINI_API_KEY", DEFAULT_LLM_MODELS["gemini"]),
        ("huggingface", "HF_TOKEN", DEFAULT_LLM_MODELS["huggingface"]),
    ]
    _say("\nAvailable cloud providers:")
    for i, (name, env, model) in enumerate(providers, 1):
        present = "Y" if os.environ.get(env) else "-"
        _say(f"  {i}. {name:12s} ({env}={present}) default_model={model}")

    choice = _ask("Choose provider number (or Enter to skip): ")
    if not choice:
        _save_ollama_cache(script_dir, {"cloud_prompted": True})
        return
    try:
        name, env, default_model = providers[int(choice) - 1]
    except (ValueError, IndexError):
        _say("Invalid choice, skipping cloud opt-in.")
        _save_ollama_cache(script_dir, {"cloud_prompted": True})
        return

    if not os.environ.get(env):
        key = _ask_secret(f"Paste {env} (hidden if supported): ")
        if not key:
            _say("No key provided, skipping cloud opt-in.")
            _save_ollama_cache(script_dir, {"cloud_prompted": True})
            return
        os.environ[env] = key.strip()

    _say(f"Validating {name}…")
    fn = dict(LLM_CHAIN).get(name)
    if fn is None:
        _say("Provider not supported, skipping.")
        _save_ollama_cache(script_dir, {"cloud_prompted": True})
        return
    test = None
    try:
        test = fn("Reply with exactly the word ok.", model=default_model, log=log)
    except Exception as e:
        if log:
            log.warn(f"cloud_validation_err provider={name} err={type(e).__name__}")
    if not test:
        retry = _ask_secret(f"Validation failed. Paste {env} again or Enter to skip: ")
        if retry:
            os.environ[env] = retry.strip()
            try:
                test = fn("Reply with exactly the word ok.", model=default_model, log=log)
            except Exception:
                test = None
    if not test:
        _say("Cloud provider validation failed. Continuing with local only.")
        _save_ollama_cache(script_dir, {"cloud_prompted": True})
        return

    config["cloud_provider"] = name
    config["cloud_model"] = default_model
    _save_ollama_cache(script_dir, {
        "cloud_prompted": True,
        "cloud_provider": name,
        "cloud_model": default_model,
    })
    _say(f"Cloud expanded: {name} model={default_model}")
    if log:
        log.info(f"cloud_opted_in provider={name} model={default_model}")


# ---------------------------------------------------------------------------
# Free public API fan-out (no keys required)
# ---------------------------------------------------------------------------
def _api_github_repos(q, timeout):
    url = ("https://api.github.com/search/repositories?q="
           + urllib.parse.quote(q) + "&sort=stars&per_page=5")
    data = http_get_json(url, timeout=timeout, headers=_github_headers())
    return [{"source": "github",
             "title": it.get("full_name"),
             "url": it.get("html_url"),
             "summary": (it.get("description") or "")[:300],
             "meta": {"stars": it.get("stargazers_count"), "lang": it.get("language")}}
            for it in data.get("items", [])]


def _wikipedia_summary(title, timeout):
    t = urllib.parse.quote(title.replace(" ", "_"))
    data = _safe_get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{t}", timeout=timeout)
    if not data or not data.get("extract"):
        return None
    return {"source": "wikipedia", "title": data.get("title", title),
            "url": (data.get("content_urls") or {}).get("desktop", {}).get("page"),
            "summary": data["extract"][:400], "meta": {}}


def _api_wikipedia(q, timeout):
    # Exact-title first, then the search API.
    hit = _wikipedia_summary(q, timeout)
    if hit:
        return [hit]
    try:
        s = http_get_json(
            "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
            "&srlimit=1&srsearch=" + urllib.parse.quote(q), timeout=timeout)
        results = (s.get("query") or {}).get("search") or []
        if results:
            return [h for h in [_wikipedia_summary(results[0]["title"], timeout)] if h]
    except Exception:
        pass
    return []


def _api_stackexchange(q, timeout):
    url = ("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&"
           "site=stackoverflow&pagesize=5&q=" + urllib.parse.quote(q))
    data = http_get_json(url, timeout=timeout)
    if not isinstance(data, dict):
        return []
    return [{"source": "stackexchange", "title": it.get("title"),
             "url": it.get("link"), "summary": "",
             "meta": {"score": it.get("score"), "answered": it.get("is_answered")}}
            for it in data.get("items", [])]


def _api_hn(q, timeout):
    url = ("https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage=5&query="
           + urllib.parse.quote(q))
    data = http_get_json(url, timeout=timeout)
    if not isinstance(data, dict):
        return []
    return [{"source": "hackernews", "title": h.get("title"),
             "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
             "summary": "",
             "meta": {"points": h.get("points"), "comments": h.get("num_comments")}}
            for h in data.get("hits", [])]


def _api_pypi(q, timeout):
    data = _safe_get_json(f"https://pypi.org/pypi/{urllib.parse.quote(q)}/json", timeout=timeout)
    if not data:
        return []
    info = data.get("info") or {}
    return [{"source": "pypi", "title": info.get("name", q),
             "url": info.get("project_url") or f"https://pypi.org/project/{q}/",
             "summary": (info.get("summary") or "")[:300],
             "meta": {"version": info.get("version")}}]


def _api_npm(q, timeout):
    data = _safe_get_json(f"https://registry.npmjs.org/{urllib.parse.quote(q)}", timeout=timeout)
    if not data:
        return []
    latest = (data.get("dist-tags") or {}).get("latest", "")
    desc = data.get("description", "")
    return [{"source": "npm", "title": data.get("name", q),
             "url": f"https://www.npmjs.com/package/{q}",
             "summary": (desc or "")[:300],
             "meta": {"version": latest}}]


def _api_crates(q, timeout):
    data = _safe_get_json(f"https://crates.io/api/v1/crates/{urllib.parse.quote(q)}",
                          timeout=timeout)
    if not data:
        return []
    c = data.get("crate", {}) or {}
    if not c:
        return []
    return [{"source": "crates", "title": c.get("name", q),
             "url": f"https://crates.io/crates/{q}",
             "summary": (c.get("description") or "")[:300],
             "meta": {"version": c.get("max_version")}}]


def _api_github_code_search(q, timeout, token=None, log=None):
    """Use the shared headers helper (auth + Accept + User-Agent) for consistency."""
    headers = _github_headers(accept="application/vnd.github.text-match+json")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = "https://api.github.com/search/code?per_page=5&q=" + urllib.parse.quote(q)
    try:
        data = http_get_json(url, timeout=timeout, headers=headers)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            if log:
                log.info(f"github_code_search_skipped reason=auth_required "
                         f"code={e.code} token_present={bool(token)}")
            return []
        raise
    return [{"source": "github-code", "title": it.get("name"),
             "url": it.get("html_url"), "summary": "",
             "meta": {"repo": (it.get("repository") or {}).get("full_name")}}
            for it in data.get("items", [])]


FREE_API_PROVIDERS = [
    ("github", _api_github_repos),
    ("wikipedia", _api_wikipedia),
    ("stackexchange", _api_stackexchange),
    ("hackernews", _api_hn),
    ("pypi", _api_pypi),
    ("npm", _api_npm),
    ("crates", _api_crates),
]

_SOURCE_WEIGHT = {
    "github": 1.5,
    "wikipedia": 1.3,
    "stackexchange": 1.2,
    "hackernews": 1.1,
    "pypi": 1.1,
    "npm": 1.1,
    "crates": 1.1,
    "github-code": 1.0,
}


def _rank_hits(hits, query):
    """Keyword-overlap ranking with per-source weight."""
    q = [t.lower() for t in re.findall(r"[A-Za-z0-9_+-]{2,}", query or "")]
    if not q:
        return hits

    scored = []
    for h in hits:
        blob = ((h.get("title") or "") + " " + (h.get("summary") or "")).lower()
        matched = [t for t in q if t in blob]
        weight = _SOURCE_WEIGHT.get(h.get("source", ""), 1.0)
        s = len(matched) * weight
        meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
        meta["_rank_reason"] = (
            f"matched={','.join(matched[:4])}" if matched
            else "no_keyword_match"
        ) + f" weight={weight}"
        h["meta"] = meta
        scored.append((s, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored]


def _apply_mmr(hits, lambda_=0.7, cap=100):
    """Maximal Marginal Relevance diversification.

    Applies to the first `cap` hits; the tail is appended unchanged so no
    results are lost.

    Complexity: O(cap^3) time because we recompute max Jaccard similarity
    against the selected set on every iteration, then scan the remaining
    hits. `cap` defaults to 100 (safety bound) so worst-case ~1e6 ops.
    """
    if len(hits) <= 1:
        return hits
    head = hits[:cap]
    tail = hits[cap:]
    n = len(head)
    relevance = {i: (n - i) / n for i in range(n)}

    def toks(h):
        blob = ((h.get("title") or "") + " " + (h.get("summary") or "")).lower()
        return set(re.findall(r"[a-z0-9_+-]{2,}", blob))

    token_sets = [toks(h) for h in head]

    def jaccard(a, b):
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    selected = []
    remaining = list(range(n))
    while remaining:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best_i = None
        best_score = -float("inf")
        for i in remaining:
            max_sim = max(jaccard(token_sets[i], token_sets[j]) for j in selected)
            score = lambda_ * relevance[i] - (1 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)
    return [head[i] for i in selected] + tail


def _fanout_cache_path(out_dir: Path, query: str) -> Path:
    h = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16]
    return out_dir / ".cache" / f"fanout_{h}.json"


def _load_fanout_cache(path: Path, ttl_s: int):
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > ttl_s:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_fanout_cache(path: Path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


def fan_out_search(query, timeout=20, log=None, extra_token=None, out_dir=None,
                   cache_ttl_s=3600, use_cache=True, mmr_lambda=0.7, mmr_cap=100):
    """Query all free public APIs in parallel. Returns ranked, MMR-diversified hits."""
    cache_path = _fanout_cache_path(out_dir, query) if (out_dir and use_cache) else None
    if cache_path is not None:
        cached = _load_fanout_cache(cache_path, cache_ttl_s)
        if cached is not None:
            if log:
                log.info(f"fanout_cache_hit query={query} hits={len(cached)}")
            return cached

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        for name, fn in FREE_API_PROVIDERS:
            futs[ex.submit(fn, query, timeout)] = name
        futs[ex.submit(_api_github_code_search, query, timeout, extra_token, log)] = "github-code"
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            try:
                results.extend(fut.result())
            except urllib.error.HTTPError as e:
                if log:
                    log.warn(f"fanout_http name={name} code={e.code}")
            except Exception as e:
                if log:
                    log.warn(f"fanout_err name={name} err={type(e).__name__}")

    ranked = _rank_hits(results, query)
    diversified = _apply_mmr(ranked, lambda_=mmr_lambda, cap=mmr_cap)

    if cache_path is not None:
        _save_fanout_cache(cache_path, diversified)
    if log:
        log.info(f"fanout_ok query={query} hits={len(diversified)} "
                 f"mmr_lambda={mmr_lambda} cap={mmr_cap}")
    return diversified


# ---------------------------------------------------------------------------
# Free-tier / paid LLM providers
# ---------------------------------------------------------------------------
def _http_post_json(url, payload, headers, timeout=60):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _llm_model(name: str, cfg: dict = None) -> str:
    env = os.environ.get(LLM_MODEL_ENV.get(name, ""))
    if env:
        return env
    if cfg:
        v = cfg.get(f"llm_model_{name}")
        if isinstance(v, str) and v.strip():
            return v
    return DEFAULT_LLM_MODELS[name]


def _llm_groq(prompt, model=None, log=None):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    model = model or _llm_model("groq")
    try:
        d = _http_post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 3000, "temperature": 0.3},
            {"Authorization": f"Bearer {key}"})
        if log:
            log.info(f"llm_call provider=groq model={model}")
        return d["choices"][0]["message"]["content"]
    except Exception as e:
        if log:
            log.warn(f"llm_call_failed provider=groq model={model} err={type(e).__name__}")
        return None


def _llm_openrouter(prompt, model=None, log=None):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    model = model or _llm_model("openrouter")
    try:
        d = _http_post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 3000},
            {"Authorization": f"Bearer {key}",
             "HTTP-Referer": "https://localhost/rea_mini", "X-Title": "REA_mini"})
        if log:
            log.info(f"llm_call provider=openrouter model={model}")
        return d["choices"][0]["message"]["content"]
    except Exception as e:
        if log:
            log.warn(f"llm_call_failed provider=openrouter model={model} err={type(e).__name__}")
        return None


def _llm_openai(prompt, model=None, log=None):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    model = model or _llm_model("openai")
    try:
        d = _http_post_json(
            "https://api.openai.com/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 3000, "temperature": 0.3},
            {"Authorization": f"Bearer {key}"})
        if log:
            log.info(f"llm_call provider=openai model={model}")
        return d["choices"][0]["message"]["content"]
    except Exception as e:
        if log:
            log.warn(f"llm_call_failed provider=openai model={model} err={type(e).__name__}")
        return None


def _llm_anthropic(prompt, model=None, log=None):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    model = model or _llm_model("anthropic")
    try:
        d = _http_post_json(
            "https://api.anthropic.com/v1/messages",
            {"model": model, "max_tokens": 3000,
             "messages": [{"role": "user", "content": prompt}]},
            {"x-api-key": key,
             "anthropic-version": "2023-06-01",
             "content-type": "application/json"})
        if log:
            log.info(f"llm_call provider=anthropic model={model}")
        blocks = d.get("content") or []
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return "\n".join(texts) if texts else None
    except Exception as e:
        if log:
            log.warn(f"llm_call_failed provider=anthropic model={model} err={type(e).__name__}")
        return None


def _llm_openai_compatible(prompt, model=None, log=None, *,
                           base_url=None, api_key_env=None,
                           provider_name="openai_compat"):
    env_name = api_key_env or "OPENAI_API_KEY"
    key = os.environ.get(env_name)
    if not key:
        return None
    base = (base_url or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    model = model or _llm_model(provider_name)
    try:
        d = _http_post_json(
            f"{base}/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 3000, "temperature": 0.3},
            {"Authorization": f"Bearer {key}"})
        if log:
            log.info(f"llm_call provider={provider_name} model={model} base={base}")
        return d["choices"][0]["message"]["content"]
    except Exception as e:
        if log:
            log.warn(f"llm_call_failed provider={provider_name} "
                     f"model={model} err={type(e).__name__}")
        return None


def _llm_deepseek(prompt, model=None, log=None):
    return _llm_openai_compatible(
        prompt, model=model, log=log,
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        provider_name="deepseek")


def _llm_mistral(prompt, model=None, log=None):
    return _llm_openai_compatible(
        prompt, model=model, log=log,
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        provider_name="mistral")


def _llm_gemini(prompt, model=None, log=None):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    model = model or _llm_model("gemini")
    try:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        d = _http_post_json(url, {"contents": [{"parts": [{"text": prompt}]}]}, {})
        if log:
            log.info(f"llm_call provider=gemini model={model}")
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        if log:
            log.warn(f"llm_call_failed provider=gemini model={model} err={type(e).__name__}")
        return None


def _llm_huggingface(prompt, model=None, log=None):
    key = os.environ.get("HF_TOKEN")
    if not key:
        return None
    model = model or _llm_model("huggingface")
    try:
        d = _http_post_json(
            f"https://api-inference.huggingface.co/models/{model}",
            {"inputs": prompt, "parameters": {"max_new_tokens": 3000}},
            {"Authorization": f"Bearer {key}"})
    except Exception as e:
        if log:
            log.warn(f"llm_call_failed provider=huggingface model={model} "
                     f"err={type(e).__name__}")
        return None
    if isinstance(d, list) and d:
        if log:
            log.info(f"llm_call provider=huggingface model={model}")
        return d[0].get("generated_text")
    if isinstance(d, dict) and d.get("error"):
        if log:
            log.warn(f"llm_call_failed provider=huggingface model={model} "
                     f"api_error={d.get('error')!r}")
        return None
    return None


def _llm_ollama(prompt, model=None, log=None):
    host = _ollama_host()
    model = model or _llm_model("ollama")

    # Prefer chat API (newer Ollama), fall back to /api/generate.
    try:
        d = _http_post_json(
            f"{host}/api/chat",
            {"model": model,
             "messages": [{"role": "user", "content": prompt}],
             "stream": False,
             "options": {"temperature": 0.3}},
            {}, timeout=180)
        msg = (d.get("message") or {}).get("content")
        if msg:
            if log:
                log.info(f"llm_call provider=ollama model={model} endpoint=chat")
            return msg
    except Exception as e:
        if log:
            log.warn(f"llm_ollama_chat_failed model={model} err={type(e).__name__}")

    try:
        d = _http_post_json(
            f"{host}/api/generate",
            {"model": model, "prompt": prompt, "stream": False},
            {}, timeout=180)
        if log:
            log.info(f"llm_call provider=ollama model={model} endpoint=generate")
        return d.get("response")
    except Exception as e:
        if log:
            log.warn(f"llm_ollama_generate_failed model={model} err={type(e).__name__}")
        return None


LLM_CHAIN = [
    ("groq", _llm_groq),
    ("openrouter", _llm_openrouter),
    ("openai", _llm_openai),
    ("anthropic", _llm_anthropic),
    ("deepseek", _llm_deepseek),
    ("mistral", _llm_mistral),
    ("gemini", _llm_gemini),
    ("huggingface", _llm_huggingface),
    ("ollama", _llm_ollama),
]


def llm_fan_out(prompt, log=None, cfg=None):
    """Try local-first when require_local_llm, cloud-second if opted in.

    Non-required mode keeps the historical chain (free-tier first, ollama last).
    """
    cfg = cfg or {}
    require_local = bool(cfg.get("require_local_llm", True))
    strict = bool(cfg.get("strict_local_llm_skip_cloud", True))
    local_model = (cfg.get("llm_fallback_model") or "").strip() or _llm_model("ollama", cfg)
    cloud_provider = (cfg.get("cloud_provider") or "").strip()
    cloud_model = (cfg.get("cloud_model") or "").strip()

    if require_local:
        # Local-first.
        try:
            text = _llm_ollama(prompt, model=local_model, log=log)
        except Exception as e:
            if log:
                log.warn(f"llm_provider_err name=ollama err={type(e).__name__}")
            text = None
        if text and text.strip():
            if log:
                log.info(f"llm_ok provider=ollama model={local_model} chars={len(text)}")
            return "ollama", text
        if log:
            log.warn("local_llm_call_failed")

        # Optional cloud fallback only if user opted in.
        if cloud_provider and not strict:
            fn = dict(LLM_CHAIN).get(cloud_provider)
            if fn:
                try:
                    text = fn(prompt, model=cloud_model or None, log=log)
                except Exception as e:
                    if log:
                        log.warn(f"llm_provider_err name={cloud_provider} "
                                 f"err={type(e).__name__}")
                    text = None
                if text and text.strip():
                    if log:
                        log.info(f"llm_ok provider={cloud_provider} chars={len(text)}")
                    return cloud_provider, text
        elif cloud_provider and strict:
            # Per strict mode we still try cloud, but only after local failed.
            fn = dict(LLM_CHAIN).get(cloud_provider)
            if fn:
                try:
                    text = fn(prompt, model=cloud_model or None, log=log)
                except Exception as e:
                    if log:
                        log.warn(f"llm_provider_err name={cloud_provider} "
                                 f"err={type(e).__name__}")
                    text = None
                if text and text.strip():
                    if log:
                        log.info(f"llm_ok provider={cloud_provider} chars={len(text)}")
                    return cloud_provider, text

        if log:
            log.info("llm_all_skipped reason=require_local_llm_and_all_failed")
        return None, None

    # Non-required mode: try the full chain (ollama last).
    for name, fn in LLM_CHAIN:
        try:
            text = fn(prompt, model=_llm_model(name, cfg), log=log)
        except Exception as e:
            if log:
                log.warn(f"llm_provider_err name={name} err={type(e).__name__}")
            continue
        if text and text.strip():
            if log:
                log.info(f"llm_ok provider={name} chars={len(text)}")
            return name, text
    if log:
        log.info("llm_all_skipped no_keys_or_ollama_down")
    return None, None


def cmd_list_models(cfg, log):
    require = bool(cfg.get("require_local_llm", True))
    local = (cfg.get("llm_fallback_model") or "").strip() or STARTER_MODEL
    cloud = (cfg.get("cloud_provider") or "").strip()
    cloud_model = (cfg.get("cloud_model") or "").strip()

    print("LLM configuration:")
    print(f"  require_local_llm: {require}")
    print(f"  local model:       {local}")
    print(f"  cloud provider:    {cloud or '<none>'}")
    print(f"  cloud model:       {cloud_model or '<none>'}")
    print()

    if require:
        print("Chain (local first; optional cloud only if opted in):")
        print(f"  1. ollama             model={local}")
        if cloud:
            print(f"  2. {cloud:18s} model={cloud_model or DEFAULT_LLM_MODELS.get(cloud, '?')}")
    else:
        print("Chain (all providers; ollama last):")
        for i, (name, _fn) in enumerate(LLM_CHAIN, 1):
            env_key = PROVIDER_KEY_ENV.get(name)
            key_set = True if env_key is None else bool(os.environ.get(env_key))
            model = _llm_model(name, cfg)
            override = ""
            if os.environ.get(LLM_MODEL_ENV.get(name, "")):
                override = " [env override]"
            elif cfg.get(f"llm_model_{name}"):
                override = " [config override]"
            status = "ready" if key_set else "no_key"
            print(f"  {i}. {name:12s} model={model:45s} {status}{override}")
    print()
    print(f"OLLAMA_HOST={os.environ.get('OLLAMA_HOST', DEFAULT_OLLAMA_HOST)}")
    print()
    print("Model override env vars (set to pin a specific model):")
    for name, env in LLM_MODEL_ENV.items():
        val = os.environ.get(env)
        print(f"  {env}={val if val else '<unset>'}")
    if log:
        log.info("list_models_shown")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Ghidra headless
# ---------------------------------------------------------------------------
GHIDRA_POSTSCRIPT_JAVA = r'''//@category REA
// Dumps functions and printable strings from the analyzed program to JSON.
// Usage: -postScript ReaDump.java /abs/path/to/dump.json
//
// Note: strings are scanned block-by-block; a printable run that spans a
// MemoryBlock boundary is emitted as two separate entries. This matches
// the semantics of "printable ASCII runs >= 4 chars within one block".
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.*;
import ghidra.program.model.mem.*;
import java.io.*;

public class ReaDump extends GhidraScript {

    private static String jsonStr(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int)c));
                    else sb.append(c);
            }
        }
        sb.append("\"");
        return sb.toString();
    }

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        String outPath = (args != null && args.length > 0) ? args[0] : "rea_dump.json";
        PrintWriter w = new PrintWriter(new FileWriter(outPath));
        try {
            w.println("{");
            w.println("  \"program\": " + jsonStr(currentProgram.getName()) + ",");
            w.println("  \"language\": " + jsonStr(currentProgram.getLanguageID().toString()) + ",");
            w.println("  \"image_base\": " + jsonStr(currentProgram.getImageBase().toString()) + ",");

            w.println("  \"functions\": [");
            FunctionManager fm = currentProgram.getFunctionManager();
            FunctionIterator fit = fm.getFunctions(true);
            boolean first = true;
            int count = 0;
            while (fit.hasNext() && count < 20000) {
                Function f = fit.next();
                if (!first) w.println(",");
                first = false;
                w.print("    {\"name\": " + jsonStr(f.getName())
                    + ", \"entry\": " + jsonStr(f.getEntryPoint().toString())
                    + ", \"size\": " + f.getBody().getNumAddresses()
                    + "}");
                count++;
            }
            w.println();
            w.println("  ],");

            w.println("  \"strings\": [");
            first = true;
            for (MemoryBlock block : currentProgram.getMemory().getBlocks()) {
                if (!block.isInitialized()) continue;
                Address start = block.getStart();
                long len = block.getSize();
                byte[] buf = new byte[4096];
                long pos = 0;
                StringBuilder sb = new StringBuilder();
                long sbStart = 0;
                while (pos < len) {
                    int n = (int)Math.min(buf.length, len - pos);
                    int read;
                    try {
                        read = block.getBytes(start.add(pos), buf, 0, n);
                    } catch (MemoryAccessException ex) {
                        break;
                    }
                    if (read <= 0) break;
                    for (int i = 0; i < read; i++) {
                        byte b = buf[i];
                        if (b >= 32 && b < 127) {
                            if (sb.length() == 0) sbStart = pos + i;
                            sb.append((char)(b & 0xff));
                        } else {
                            if (sb.length() >= 4) {
                                if (!first) w.println(",");
                                first = false;
                                w.print("    {\"addr\": " + jsonStr(start.add(sbStart).toString())
                                    + ", \"text\": " + jsonStr(sb.toString()) + "}");
                            }
                            sb.setLength(0);
                        }
                    }
                    pos += read;
                }
                if (sb.length() >= 4) {
                    if (!first) w.println(",");
                    first = false;
                    w.print("    {\"addr\": " + jsonStr(start.add(sbStart).toString())
                        + ", \"text\": " + jsonStr(sb.toString()) + "}");
                }
            }
            w.println();
            w.println("  ]");
            w.println("}");
        } finally {
            w.close();
        }
        println("REA dump written to " + outPath);
    }
}
'''


def _ensure_ghidra_postscript(script_dir: Path, log) -> Path:
    scripts_dir = script_dir / "ghidra_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    p = scripts_dir / "ReaDump.java"
    try:
        if not p.exists() or p.read_text(encoding="utf-8") != GHIDRA_POSTSCRIPT_JAVA:
            atomic_write_text(p, GHIDRA_POSTSCRIPT_JAVA)
            log.info(f"ghidra_postscript_written path={p}")
    except OSError as e:
        log.warn(f"ghidra_postscript_write_failed err={e}")
    return scripts_dir


def ghidra_headless_run(ghidra_path: Path, binary: Path, project_dir: Path,
                        script_dir: Path, keywords=None, log=None, timeout_s=300):
    """Run analyzeHeadless with our ReaDump postScript; return structured result.

    Strings for keyword scans come from the postScript dump first; external
    `strings` is used only as a fallback if the dump is empty. The
    `-analysisTimeoutPerFile` flag is only passed if the installed
    analyzeHeadless advertises support, which is cached in ghidra_cache.json.
    """
    analyze = _analyze_headless_path(ghidra_path)
    if not analyze:
        if log:
            log.error("ghidra_headless_missing analyzeHeadless")
        return {"error": "analyzeHeadless missing"}

    project_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = _ensure_ghidra_postscript(script_dir, log)
    dump_path = project_dir / f"rea_dump_{int(time.time())}.json"
    proj_name = f"rea_{int(time.time())}"

    cache = _load_ghidra_cache(script_dir)
    if "supports_analysis_timeout" in cache:
        supports_timeout = bool(cache["supports_analysis_timeout"])
    else:
        supports_timeout = check_ghidra_timeout_flag(ghidra_path)
        cache["supports_analysis_timeout"] = supports_timeout
        _save_ghidra_cache(script_dir, cache)

    cmd = [str(analyze), str(project_dir), proj_name,
           "-import", str(binary),
           "-scriptPath", str(scripts_dir)]
    if supports_timeout:
        cmd.extend(["-analysisTimeoutPerFile", str(timeout_s)])
    cmd.extend(["-postScript", "ReaDump.java", str(dump_path)])
    cmd.append("-deleteProject")

    if log:
        log.info(f"ghidra_run cmd={cmd[0]} binary={binary} timeout_flag={supports_timeout}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 180)
    except subprocess.TimeoutExpired:
        if log:
            log.error("ghidra_run_failed reason=timeout")
        return {"error": "timeout"}
    except Exception as e:
        if log:
            log.error(f"ghidra_run_failed err={e}")
        return {"error": str(e)}

    out = {"stdout": (r.stdout or "")[-4000:], "stderr": (r.stderr or "")[-2000:],
           "rc": r.returncode, "postscript_used": True,
           "timeout_flag_supported": supports_timeout}

    structured = None
    if dump_path.exists():
        try:
            with open(dump_path, "r", encoding="utf-8") as f:
                structured = json.load(f)
            dump_path.unlink(missing_ok=True)
            if log:
                log.info(f"ghidra_dump_ok functions={len(structured.get('functions', []))} "
                         f"strings={len(structured.get('strings', []))}")
        except (OSError, json.JSONDecodeError) as e:
            if log:
                log.warn(f"ghidra_dump_parse_failed err={type(e).__name__}")
    else:
        if log:
            log.warn("ghidra_dump_missing postscript_did_not_write")
        out["postscript_used"] = False

    out["dump"] = structured

    # Keyword scan: authoritative source is the dump; strings tool is the
    # fallback. Warn exactly once if `strings` is unavailable and we needed it.
    hits = {}
    if keywords:
        strings_pool = []
        if structured and isinstance(structured.get("strings"), list):
            strings_pool = [s.get("text", "") for s in structured["strings"] if s.get("text")]
        if not strings_pool:
            if shutil.which("strings"):
                try:
                    strings_out = subprocess.run(
                        ["strings", "-a", "-n", "4", str(binary)],
                        capture_output=True, text=True, timeout=60).stdout
                    strings_pool = strings_out.splitlines()
                except Exception as e:
                    if log:
                        log.warn(f"strings_failed err={type(e).__name__}")
            else:
                if log:
                    log.warn("strings_missing reason=binutils_not_installed "
                             "fallback=dump_only")
                print("Note: 'strings' (binutils) not found; keyword scan uses the "
                      "postScript dump only. Install: sudo apt install binutils")

        for kw in keywords:
            kws = kw.lower()
            hits[kw] = [ln for ln in strings_pool if kws in ln.lower()][:20]
    out["keyword_hits"] = hits
    if log:
        log.info(f"ghidra_run_ok rc={r.returncode} keywords={len(hits)}")
    return out


# ---------------------------------------------------------------------------
# Skeleton generators (deterministic, functional templates)
# ---------------------------------------------------------------------------
def _meta_line(idea, extra=""):
    ts = datetime.now(timezone.utc).isoformat()
    base = f"Generated by REA_mini.py at {ts} | Idea: {idea.replace(chr(10), ' | ')}"
    return (base + (" | " + extra if extra else "")).rstrip()


def _safe_py_docstring(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _safe_comment(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ")


_TMPL_PY = string.Template(r'''#!/usr/bin/env python3
"""
$idea_doc
"""
# $meta
import argparse
import hashlib
import json
import sys


def run_pipeline(text: str) -> dict:
    """Minimal working core: normalise + hash + report basic stats.

    `length` is the number of Unicode characters in `text`.
    """
    if not text:
        return {"status": "no_input"}
    return {
        "status": "ok",
        "length": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "upper": text.upper(),
        "words": len(text.split()),
    }


def self_test() -> None:
    r = run_pipeline("abc")
    assert r["status"] == "ok" and r["length"] == 3, r
    assert run_pipeline("")["status"] == "no_input"
    print("self-tests: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=$desc)
    p.add_argument("positional", nargs="*", help="text to process")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--test", action="store_true", help="run self-tests and exit")
    args = p.parse_args(argv)
    if args.test:
        self_test()
        return 0
    result = run_pipeline(" ".join(args.positional))
    text = json.dumps(result, indent=2)
    print(text, file=sys.stderr if args.verbose else sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
''')

_TMPL_JS = string.Template(r'''#!/usr/bin/env node
// $meta
'use strict';

const crypto = require('crypto');

function runPipeline(text) {
  // `length` is the JS code-unit length (UTF-16).
  if (!text) return { status: 'no_input' };
  return {
    status: 'ok',
    length: text.length,
    sha256: crypto.createHash('sha256').update(text, 'utf8').digest('hex'),
    upper: text.toUpperCase(),
    words: text.trim().split(/\s+/).filter(Boolean).length,
  };
}

function selfTest() {
  const a = runPipeline('abc');
  if (a.status !== 'ok' || a.length !== 3) throw new Error('test 1 failed');
  if (runPipeline('').status !== 'no_input') throw new Error('test 2 failed');
  console.log('self-tests: ok');
}

function main(argv) {
  const args = { verbose: false, test: false, positional: [] };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--verbose' || a === '-v') args.verbose = true;
    else if (a === '--test') args.test = true;
    else args.positional.push(a);
  }
  if (args.test) { selfTest(); return 0; }
  const result = runPipeline(args.positional.join(' '));
  const out = JSON.stringify(result, null, 2);
  if (args.verbose) console.error(out); else console.log(out);
  return 0;
}

process.exit(main(process.argv));
''')

_TMPL_HTML = string.Template(r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<!-- $meta -->
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>$title_short</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.5 system-ui, sans-serif; max-width: 960px;
         margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; font: 14px/1.4 ui-monospace, monospace; min-height: 6rem; }
  button { padding: .5rem 1rem; font: inherit; margin-top: .5rem; }
  pre { background: #111; color: #eee; padding: 1rem; overflow: auto; }
</style>
</head>
<body>
<header>
  <h1>$idea_html</h1>
</header>
<main>
  <label for="in">Input</label>
  <textarea id="in" placeholder="type text to process"></textarea>
  <button id="run" type="button">Run</button>
  <button id="test" type="button">Run self-test</button>
  <pre id="out">(no output yet)</pre>
</main>
<script>
  function hasCryptoSubtle() {
    return typeof crypto !== 'undefined'
      && crypto.subtle
      && typeof crypto.subtle.digest === 'function';
  }
  async function sha256(text) {
    if (hasCryptoSubtle()) {
      const data = new TextEncoder().encode(text);
      const buf = await crypto.subtle.digest('SHA-256', data);
      return Array.from(new Uint8Array(buf))
        .map(b => b.toString(16).padStart(2, '0')).join('');
    }
    // Fallback: FNV-1a (NOT cryptographic; used only when crypto.subtle is
    // unavailable, e.g. non-secure contexts / old Safari).
    let h = 0x811c9dc5;
    const data = new TextEncoder().encode(text);
    for (let i = 0; i < data.length; i++) {
      h ^= data[i];
      h = (h * 0x01000193) >>> 0;
    }
    return 'fnv1a-' + h.toString(16).padStart(8, '0');
  }
  async function runPipeline(text) {
    if (!text) return { status: 'no_input' };
    return {
      status: 'ok',
      length: text.length,
      sha256: await sha256(text),
      upper: text.toUpperCase(),
      words: text.trim().split(/\s+/).filter(Boolean).length,
    };
  }
  document.getElementById('run').addEventListener('click', async () => {
    const text = document.getElementById('in').value;
    const out = document.getElementById('out');
    try { out.textContent = JSON.stringify(await runPipeline(text), null, 2); }
    catch (e) { out.textContent = 'error: ' + (e && e.message ? e.message : e); }
  });
  document.getElementById('test').addEventListener('click', async () => {
    const out = document.getElementById('out');
    try {
      const a = await runPipeline('abc');
      if (a.status !== 'ok' || a.length !== 3) throw new Error('test 1');
      if ((await runPipeline('')).status !== 'no_input') throw new Error('test 2');
      out.textContent = 'self-tests: ok';
    } catch (e) { out.textContent = 'self-test failed: ' + (e && e.message ? e.message : e); }
  });
</script>
</body>
</html>
''')

_TMPL_RUST = string.Template(r'''// $meta
// Build: rustc main.rs -o app
//
// `length` is the number of Unicode scalar values (chars) in the input.
use std::env;
use std::process::ExitCode;

fn run_pipeline(text: &str) -> String {
    if text.is_empty() {
        return String::from("{\"status\":\"no_input\"}");
    }
    let upper = text.to_uppercase();
    let words = text.split_whitespace().count();
    format!(
        "{{\"status\":\"ok\",\"length\":{},\"upper\":\"{}\",\"words\":{}}}",
        text.chars().count(),
        upper.replace('"', "\\\""),
        words
    )
}

fn self_test() {
    assert!(run_pipeline("abc").contains("\"length\":3"));
    assert!(run_pipeline("").contains("no_input"));
    println!("self-tests: ok");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--test") {
        self_test();
        return ExitCode::SUCCESS;
    }
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let text = args.iter().filter(|a| !a.starts_with("--") && *a != "-v").cloned()
        .collect::<Vec<_>>().join(" ");
    let out = run_pipeline(&text);
    if verbose { eprintln!("{}", out); } else { println!("{}", out); }
    ExitCode::SUCCESS
}
''')

_TMPL_CPP = string.Template(r'''// $meta
// Build: g++ -std=c++20 -O2 -o app main.cpp
//
// `length` is the number of UTF-8 code points in the input (matches the
// Python/Rust "character count" semantics).
#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <sstream>

namespace rea {

struct Options {
    bool verbose = false;
    bool test = false;
    std::vector<std::string> positional;
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--test") o.test = true;
        else o.positional.push_back(a);
    }
    return o;
}

static size_t count_utf8_codepoints(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++n;  // not a UTF-8 continuation byte
    }
    return n;
}

std::string run_pipeline(const std::string& text) {
    if (text.empty()) return "{\"status\":\"no_input\"}";
    std::string upper;
    upper.reserve(text.size());
    for (char c : text)
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    size_t words = 0;
    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) ++words;
    std::string esc;
    esc.reserve(upper.size() * 2);
    for (char c : upper) {
        if (c == '"' || c == '\\') esc.push_back('\\');
        esc.push_back(c);
    }
    std::ostringstream oss;
    oss << "{\"status\":\"ok\",\"length\":" << count_utf8_codepoints(text)
        << ",\"upper\":\"" << esc << "\",\"words\":" << words << "}";
    return oss.str();
}

int self_test() {
    if (run_pipeline("abc").find("\"length\":3") == std::string::npos) {
        std::fprintf(stderr, "self-test 1 failed\n");
        return 1;
    }
    if (run_pipeline("").find("no_input") == std::string::npos) {
        std::fprintf(stderr, "self-test 2 failed\n");
        return 1;
    }
    std::printf("self-tests: ok\n");
    return 0;
}

}  // namespace rea

int main(int argc, char** argv) {
    auto opts = rea::parse_args(argc, argv);
    if (opts.test) return rea::self_test();
    std::string joined;
    for (size_t i = 0; i < opts.positional.size(); ++i) {
        if (i) joined.push_back(' ');
        joined += opts.positional[i];
    }
    auto result = rea::run_pipeline(joined);
    if (opts.verbose) std::fprintf(stderr, "%s\n", result.c_str());
    else std::printf("%s\n", result.c_str());
    return 0;
}
''')


def gen_python(idea, extra=""):
    return _TMPL_PY.substitute(
        idea_doc=_safe_py_docstring(idea),
        meta=_safe_comment(_meta_line(idea, extra)),
        desc=repr(idea),
    )


def gen_js(idea, extra=""):
    return _TMPL_JS.substitute(meta=_safe_comment(_meta_line(idea, extra)))


def gen_html(idea, extra=""):
    return _TMPL_HTML.substitute(
        meta=_safe_comment(_meta_line(idea, extra)).replace("--", "-"),
        title_short=_html_escape(idea[:80]),
        idea_html=_html_escape(idea),
    )


def gen_rust(idea, extra=""):
    return _TMPL_RUST.substitute(meta=_safe_comment(_meta_line(idea, extra)))


def gen_cpp(idea, extra=""):
    return _TMPL_CPP.substitute(meta=_safe_comment(_meta_line(idea, extra)))


GENERATORS = {
    "python": (gen_python, "py"),
    "js": (gen_js, "js"),
    "html": (gen_html, "html"),
    "rust": (gen_rust, "rs"),
    "cpp": (gen_cpp, "cpp"),
}

_FIRST_TOKEN = {
    "python": ("import ", "from ", '"""', "'''", "# ", "#!/"),
    "js": ("'use strict'", "//", "const ", "let ", "var ", "function ",
           "async ", "import ", "export ", "#!/"),
    "html": ("<!doctype", "<!DOCTYPE", "<html", "<!--", "<meta"),
    "rust": ("//", "fn ", "use ", "#!", "pub ", "mod "),
    "cpp": ("//", "#include", "#pragma", "/*", "using ", "namespace "),
}


def _validate_skeleton(lang, text):
    stripped = (text or "").lstrip()
    if stripped.startswith("#!"):
        nl = stripped.find("\n")
        stripped = stripped[nl + 1:].lstrip() if nl >= 0 else ""
    tokens = _FIRST_TOKEN.get(lang, ())
    return any(stripped.startswith(t) for t in tokens)


def _extract_code_block(text, lang=None):
    """Return the best fenced block."""
    text = text or ""
    pattern = re.compile(r"```([A-Za-z0-9_+\-]*)\r?\n(.*?)\r?\n```", re.DOTALL)
    blocks = pattern.findall(text)
    if blocks:
        if lang:
            for tag, code in blocks:
                if tag.lower() == lang.lower():
                    return code
        for tag, code in blocks:
            if tag:
                return code
        return blocks[0][1]
    return text.strip()


# --- Skeleton checks ------------------------------------------------------
def _syntax_check_skeleton(lang, path: Path, log=None):
    """Lightweight syntax check — does NOT execute the generated artifact.

    Safe by default; the heavier `_smoke_test_skeleton` (which runs the
    artifact's own --test mode) is only used when the user passes
    --run-generated or explicitly opts in at the prompt.
    """
    result = {"lang": lang, "path": str(path), "executed": False,
              "passed": False, "stdout": "", "stderr": "", "returncode": None,
              "mode": "syntax"}
    try:
        if lang == "python":
            r = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                               capture_output=True, text=True, timeout=30)
        elif lang == "js":
            if not shutil.which("node"):
                result["stderr"] = "node not installed (syntax check skipped)"
                return result
            r = subprocess.run(["node", "--check", str(path)],
                               capture_output=True, text=True, timeout=30)
        elif lang == "rust":
            if not shutil.which("rustc"):
                result["stderr"] = "rustc not installed (syntax check skipped)"
                return result
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run(
                    ["rustc", "--edition", "2021", "--emit=metadata",
                     str(path), "-o", str(Path(td) / "out.rmeta")],
                    capture_output=True, text=True, timeout=120)
        elif lang == "cpp":
            if not shutil.which("g++"):
                result["stderr"] = "g++ not installed (syntax check skipped)"
                return result
            r = subprocess.run(["g++", "-std=c++20", "-fsyntax-only", str(path)],
                               capture_output=True, text=True, timeout=60)
        elif lang == "html":
            text = path.read_text(encoding="utf-8")
            has_doctype = "<!doctype" in text.lower()
            has_html = "<html" in text.lower()
            has_script = "<script" in text.lower()
            result.update(executed=True,
                          passed=(has_doctype and has_html and has_script),
                          stdout=f"doctype={has_doctype} html={has_html} script={has_script}")
            return result
        else:
            result["stderr"] = f"no syntax check for lang={lang}"
            return result

        result.update(executed=True, passed=(r.returncode == 0),
                      stdout=(r.stdout or "")[-2000:],
                      stderr=(r.stderr or "")[-2000:],
                      returncode=r.returncode)
    except subprocess.TimeoutExpired:
        result.update(executed=True, passed=False, stderr="timeout")
    except Exception as e:
        result["stderr"] = f"{type(e).__name__}: {e}"
    if log:
        log.info(f"skeleton_syntax lang={lang} executed={result['executed']} "
                 f"passed={result['passed']}")
    return result


def _smoke_test_skeleton(lang, path: Path, log=None):
    """Run the generated artifact's own --test mode. Only used under
    --run-generated or an explicit interactive confirmation, because it
    actually executes LLM-generated code."""
    result = {"lang": lang, "path": str(path), "executed": False,
              "passed": False, "stdout": "", "stderr": "", "returncode": None,
              "mode": "smoke"}
    try:
        if lang == "python":
            r = subprocess.run([sys.executable, str(path), "--test"],
                               capture_output=True, text=True, timeout=60)
            result.update(executed=True, passed=(r.returncode == 0),
                          stdout=(r.stdout or "")[-2000:],
                          stderr=(r.stderr or "")[-2000:],
                          returncode=r.returncode)
        elif lang == "js":
            if not shutil.which("node"):
                result["stderr"] = "node not installed"
                return result
            r = subprocess.run(["node", str(path), "--test"],
                               capture_output=True, text=True, timeout=60)
            result.update(executed=True, passed=(r.returncode == 0),
                          stdout=(r.stdout or "")[-2000:],
                          stderr=(r.stderr or "")[-2000:],
                          returncode=r.returncode)
        elif lang == "rust":
            if not shutil.which("rustc"):
                result["stderr"] = "rustc not installed"
                return result
            with tempfile.TemporaryDirectory() as td:
                exe = Path(td) / "app"
                rc = subprocess.run(
                    ["rustc", "--edition", "2021", str(path), "-o", str(exe)],
                    capture_output=True, text=True, timeout=120)
                if rc.returncode != 0:
                    result.update(executed=True, passed=False,
                                  stderr=f"compile failed: {(rc.stderr or '')[-1500:]}")
                    return result
                r = subprocess.run([str(exe), "--test"], capture_output=True,
                                   text=True, timeout=30)
                result.update(executed=True, passed=(r.returncode == 0),
                              stdout=(r.stdout or "")[-2000:],
                              stderr=(r.stderr or "")[-2000:],
                              returncode=r.returncode)
        elif lang == "cpp":
            if not shutil.which("g++"):
                result["stderr"] = "g++ not installed"
                return result
            with tempfile.TemporaryDirectory() as td:
                exe = Path(td) / "app"
                rc = subprocess.run(["g++", "-std=c++20", "-O2", str(path), "-o", str(exe)],
                                    capture_output=True, text=True, timeout=120)
                if rc.returncode != 0:
                    result.update(executed=True, passed=False,
                                  stderr=f"compile failed: {(rc.stderr or '')[-1500:]}")
                    return result
                r = subprocess.run([str(exe), "--test"], capture_output=True,
                                   text=True, timeout=30)
                result.update(executed=True, passed=(r.returncode == 0),
                              stdout=(r.stdout or "")[-2000:],
                              stderr=(r.stderr or "")[-2000:],
                              returncode=r.returncode)
        elif lang == "html":
            text = path.read_text(encoding="utf-8")
            has_doctype = "<!doctype" in text.lower()
            has_html = "<html" in text.lower()
            has_script = "<script" in text.lower()
            result.update(executed=True,
                          passed=(has_doctype and has_html and has_script),
                          stdout=f"doctype={has_doctype} html={has_html} script={has_script}")
        else:
            result["stderr"] = f"no smoke test for lang={lang}"
    except subprocess.TimeoutExpired:
        result.update(executed=True, passed=False, stderr="timeout")
    except Exception as e:
        result["stderr"] = f"{type(e).__name__}: {e}"
    if log:
        log.info(f"skeleton_smoke lang={lang} executed={result['executed']} "
                 f"passed={result['passed']} stderr={(result['stderr'] or '')[:120]}")
    return result


def synthesize_skeleton(lang, idea, search_hits=None, log=None, cfg=None):
    """Ask LLM for a working core; fall back to deterministic templates
    only when require_local_llm is disabled. Returns (provider, code) or
    (None, None) when the local LLM is required but no LLM is available."""
    cfg = cfg or {}
    require_local = bool(cfg.get("require_local_llm", True))

    if search_hits:
        ctx = "\n".join(
            f"- [{h['source']}] {h.get('title','')} {h.get('url','')} :: "
            f"{h.get('summary','')[:200]}"
            for h in search_hits[:15]
        )
        prompt = (
            f"Write a complete, runnable {lang} program that RECONSTRUCTS the observed "
            f"capability described below using the research context. Requirements:\n"
            f"- The program must actually run and produce output; no empty TODO stubs.\n"
            f"- Include argument parsing, a main entry point, and a working core that "
            f"implements the described capability.\n"
            f"- Include at least 2 inline self-tests, or a --test mode, that exercise the core.\n"
            f"- Use the research context for naming, APIs, and structure.\n"
            f"- Include a header comment with the idea and generation metadata.\n"
            f"- Return ONLY a single fenced code block containing the complete program.\n\n"
            f"Idea: {idea}\n\nResearch:\n{ctx}\n"
        )
    else:
        prompt = (
            f"Write a complete, runnable {lang} program that reconstructs the idea below. "
            f"Requirements:\n"
            f"- Must run and produce output; no empty TODO stubs.\n"
            f"- Argument parsing + main entry + working core.\n"
            f"- At least 2 inline self-tests, or a --test mode.\n"
            f"- Return ONLY a single fenced code block.\n\n"
            f"Idea: {idea}\n"
        )

    provider, text = llm_fan_out(prompt, log=log, cfg=cfg)
    if text:
        code = _extract_code_block(text, lang)
        if code and _validate_skeleton(lang, code):
            return provider, code
        if log:
            log.info(f"llm_output_rejected reason=format_mismatch "
                     f"provider={provider} lang={lang}")

    if require_local:
        if log:
            log.info("synthesize_skeleton_failed reason=require_local_llm")
        return None, None

    fn, _ = GENERATORS[lang]
    return "template", fn(idea)


# ---------------------------------------------------------------------------
# URL capture (llm / archive)
# ---------------------------------------------------------------------------
async def render_page(url, timeout_ms, user_agent, log,
                      wait_selector=None, scroll=False):
    log.info(f"render_start timeout_ms={timeout_ms} scroll={scroll} "
             f"wait_selector={wait_selector or '-'}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=user_agent)
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except PWTimeout as e:
                log.error(f"nav_failed reason=nav_timeout err={e}")
                return None, None
            except PWError as e:
                log.error(f"nav_failed reason=nav_error err={type(e).__name__}")
                return None, None

            if scroll:
                prev = -1
                for _ in range(10):
                    try:
                        await page.evaluate(
                            "window.scrollBy(0, Math.max(document.body.scrollHeight, "
                            "document.documentElement.scrollHeight))")
                    except Exception:
                        break
                    await page.wait_for_timeout(300)
                    try:
                        cur = await page.evaluate(
                            "Math.max(document.body.scrollHeight, "
                            "document.documentElement.scrollHeight)")
                    except Exception:
                        cur = prev
                    if cur == prev:
                        break
                    prev = cur

            if wait_selector:
                try:
                    await page.wait_for_selector(wait_selector, timeout=timeout_ms)
                except PWTimeout:
                    log.warn(f"wait_selector_timeout selector={wait_selector}")

            try:
                await page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except PWTimeout:
                log.info("networkidle_timeout continuing_with_current_dom")

            final_url = page.url
            html = await page.content()
            log.info(f"render_ok final_url={redact_url(final_url)} bytes={len(html)}")
            return final_url, html
        finally:
            try:
                await browser.close()
            except Exception:
                pass


async def _capture_with_retry(norm, config, log):
    """Two attempts: current config, then forced scroll + longer timeout."""
    timeout_s = _cfg_int(config, "timeout", 30, log)
    attempts = [
        {
            "timeout": timeout_s * 1000,
            "scroll": bool(config.get("scroll_to_bottom", False)),
            "wait_selector": _cfg_str(config, "wait_selector", "") or None,
        },
        {
            "timeout": max(timeout_s * 1500, 45000),
            "scroll": True,
            "wait_selector": None,
        },
    ]
    for i, a in enumerate(attempts):
        if log:
            log.info(f"capture_attempt n={i + 1} timeout={a['timeout']} "
                     f"scroll={a['scroll']}")
        try:
            final_url, html = await render_page(
                norm, a["timeout"], config["user_agent"], log,
                wait_selector=a["wait_selector"], scroll=a["scroll"])
        except Exception as e:
            if log:
                log.warn(f"capture_attempt_err n={i + 1} err={type(e).__name__}")
            continue
        if html:
            return final_url, html
    return None, None


def _urllib_capture(norm, config, log):
    try:
        req = urllib.request.Request(norm, headers={"User-Agent": config["user_agent"]})
        with urllib.request.urlopen(
                req, timeout=_cfg_int(config, "http_timeout", 20, log)) as r:
            final_url = r.geturl()
            raw = r.read()
        try:
            html = raw.decode("utf-8")
        except UnicodeDecodeError:
            html = raw.decode("latin-1", errors="replace")
        if log:
            log.info(f"urllib_capture_ok url={redact_url(final_url)} bytes={len(html)}")
        return final_url, html
    except Exception as e:
        if log:
            log.error(f"urllib_capture_failed err={type(e).__name__}")
        return None, None


def extract_for_llm(full_html, base_url, ignore_images, log):
    try:
        doc = _Doc(full_html)
        article = doc.summary()
        title = (doc.title() or "Untitled").strip()
    except Exception as e:
        log.error(f"extract_failed reason=readability err={type(e).__name__}")
        return None
    try:
        h = _h2t.HTML2Text()
        h.ignore_links = False
        h.ignore_images = ignore_images
        h.body_width = 0
        body = h.handle(article).strip()
    except Exception as e:
        log.error(f"extract_failed reason=html2text err={type(e).__name__}")
        return None
    if not body:
        log.error("extract_failed reason=extract_empty")
        return None
    ts = datetime.now(timezone.utc).isoformat()
    return (f"# {title}\n\n**Source:** {base_url}\n**Extracted:** {ts}\n"
            f"**Tool:** REA_mini.py\n\n---\n\n" + body)


def run_archive(url, output_path, log, cli_version="2.0.75", timeout_s=600):
    log.info(f"archive_start tool=single-file-cli version={cli_version} "
             f"timeout_s={timeout_s}")
    tmp = output_path.with_name(output_path.name + ".tmp")
    cmd = ["npx", "--yes", f"single-file-cli@{cli_version}", url, str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.error("archive_failed reason=timeout")
        return False
    except FileNotFoundError:
        log.error("archive_failed reason=npx_missing")
        return False
    except OSError as e:
        log.error(f"archive_failed reason=spawn err={e}")
        return False
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-8:]
        log.block("ERROR", [f"archive_failed reason=nonzero_exit code={r.returncode}"] + tail)
        return False
    if not tmp.exists() or tmp.stat().st_size == 0:
        log.error("archive_failed reason=no_output")
        return False
    try:
        os.replace(tmp, output_path)
    except OSError as e:
        log.error(f"archive_failed reason=rename err={e}")
        return False
    log.info(f"archive_ok output={output_path} bytes={output_path.stat().st_size}")
    return True


# ---------------------------------------------------------------------------
# Startup wizard helpers
# ---------------------------------------------------------------------------
def _ask(prompt, default=""):
    v = input(prompt).strip()
    return v if v else default


def _ask_secret(prompt):
    """Prefer getpass; fall back to plain input for non-TTY environments."""
    try:
        import getpass
        return getpass.getpass(prompt).strip()
    except Exception:
        return _ask(prompt)


def _llm_status_line(config=None):
    """One-line summary shown at the top of the interactive menu."""
    parts = []
    cfg = config or {}
    require = bool(cfg.get("require_local_llm", True))
    local = (cfg.get("llm_fallback_model") or "").strip() or STARTER_MODEL
    try:
        ok, _ = _ollama_health(timeout=1)
    except Exception:
        ok = False
    parts.append(f"local:{'Y' if ok else '-'} ({local})")
    cloud = (cfg.get("cloud_provider") or "").strip()
    parts.append(f"cloud:{cloud or '-'}")
    if not require:
        parts.append("mode:optional")
    return " ".join(parts)


def _print_startup_banner(script_dir, config):
    local = (config.get("llm_fallback_model") or "").strip() or STARTER_MODEL
    require = bool(config.get("require_local_llm", True))
    cloud = (config.get("cloud_provider") or "").strip()
    cloud_model = (config.get("cloud_model") or "").strip()
    try:
        ollama_ok, _ = _ollama_health(timeout=1)
    except Exception:
        ollama_ok = False
    try:
        gh = find_ghidra(script_dir)
        ghidra_status = "ready" if gh else "on-demand"
    except Exception:
        ghidra_status = "unknown"

    parts = []
    if require:
        parts.append(f"local={'ok' if ollama_ok else 'down'}:{local}")
    else:
        parts.append(f"local=optional:{local}")
    parts.append(f"cloud={cloud + ':' + cloud_model if cloud else 'none'}")
    parts.append(f"ghidra={ghidra_status}")
    print("REA_mini ready — " + " | ".join(parts))


def _missing_breakdown(report):
    checks = report["checks"]
    missing_pips = [p for p in ("playwright", "readability-lxml", "html2text")
                    if checks[p] != "ok"]
    missing_nonpip = []
    if not checks["chromium"]["ok"]:
        missing_nonpip.append("chromium")
    if checks["node/npx"] != "ok":
        missing_nonpip.append("node/npm")
    if checks["strings"] != "ok":
        missing_nonpip.append("binutils")
    if not checks["java21+"]["ok"]:
        missing_nonpip.append("openjdk-21")
    return missing_pips, missing_nonpip


async def _audit_dependencies_on_startup(script_dir, log, config, args):
    """Check deps; offer a single "[A]ll / [N]one / [I]nteractive" prompt."""
    report = await _gather_selfcheck(script_dir)
    missing_pips, missing_nonpip = _missing_breakdown(report)
    if not missing_pips and not missing_nonpip:
        _say("All dependencies present.")
        return True
    summary = []
    if missing_pips:
        summary.append("pip: " + ", ".join(missing_pips))
    if missing_nonpip:
        summary.append("system: " + ", ".join(missing_nonpip))
    _say("Missing dependencies — " + " | ".join(summary))

    if args.yes:
        ans = "a"
    else:
        ans = _ask("Install all? [A/n/i(nteractive)]: ", "a").lower()
    if ans in ("n", "no"):
        _say("Skipping dependency install (some features may be degraded).")
        return False
    if ans in ("i", "interactive"):
        rc = await cmd_selfcheck(script_dir, log, non_interactive=False, auto_install=False)
        return rc == EXIT_OK
    rc = await cmd_selfcheck(script_dir, log, non_interactive=True, auto_install=True)
    return rc == EXIT_OK


async def _startup_wizard(script_dir, out_dir, config, log, args):
    # Step 1: local LLM
    ok, reason = await _ensure_local_llm(
        script_dir, config, log,
        interactive=True,
        auto_pull=bool(args.yes))
    if not ok:
        _say(f"Local LLM unavailable: {reason}")
        if args.yes:
            return False
        ans = _ask("Continue without a local LLM? (cloud/template fallback) [y/N]: ",
                   "n").lower()
        if ans not in ("y", "yes"):
            return False
        config["require_local_llm"] = False

    # Step 2: dependency audit
    await _audit_dependencies_on_startup(script_dir, log, config, args)

    # Step 3: cloud expansion (only when local model is in use)
    if config.get("require_local_llm", True):
        await _cloud_opt_in(script_dir, config, log)

    # Step 4: status banner
    _print_startup_banner(script_dir, config)
    return True


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------
def _print_menu(llm_status_line=None):
    print()
    print("=" * 60)
    print(" REA_mini.py — reverse-engineer-anything mini harness")
    print("=" * 60)
    if llm_status_line:
        print(f" LLM: {llm_status_line}")
    print("  1. Capture URL -> clean Markdown")
    print("  2. Capture URL -> self-contained HTML archive")
    print("  3. Search fan-out (free public APIs)")
    print("  4. Ghidra headless analysis (binary + keywords)")
    print("  5. Generate skeleton file (HTML/JS/Py/Rust/C++)")
    print("  6. Full pipeline (URL -> fan-out -> synth -> skeleton)")
    print("  7. Self-check dependencies")
    print("  0. Quit")
    print("=" * 60)


async def cmd_capture(url, mode, output_dir, config, log):
    if not (url or "").strip():
        print("No URL provided.")
        return EXIT_BAD_URL
    norm = normalize_url(url)
    if not norm:
        print(f"Invalid URL: {url}")
        return EXIT_BAD_URL
    out = None
    if mode == "llm":
        if not (READABILITY_OK and HTML2TEXT_OK):
            print("Missing deps for llm mode (need readability-lxml + html2text). Run option 7.")
            return EXIT_MISSING_DEPS
        final_url, page_html = None, None
        used_playwright = False
        if PLAYWRIGHT_OK:
            final_url, page_html = await _capture_with_retry(norm, config, log)
            used_playwright = page_html is not None
        if page_html is None:
            _say("Playwright path failed; falling back to static fetch (urllib + readability)...")
            final_url, page_html = _urllib_capture(norm, config, log)
        if page_html is None:
            return EXIT_NAV_FAIL
        md = extract_for_llm(page_html, final_url, bool(config.get("ignore_images")), log)
        if md is None:
            return EXIT_EXTRACT_EMPTY
        out = output_dir / slugify(norm, "llm")
        try:
            atomic_write_text(out, md)
        except OSError:
            return EXIT_WRITE_FAIL
        ev = evidence_wrap(
            claim=f"Extracted LLM-friendly Markdown from {redact_url(norm)}",
            evidence=[{
                "artifact": str(out),
                "source_url": redact_url(norm),
                "final_url": redact_url(final_url),
                "bytes": out.stat().st_size,
                "tool": ("playwright+readability+html2text" if used_playwright
                         else "urllib+readability+html2text"),
                "used_playwright": used_playwright,
            }],
            limitations=[
                "Heavy SPAs, paywalls, and JS-only content may be truncated or missing.",
                "readability may discard nav/sidebar content that is contextually important.",
            ],
            unknowns=[
                "Whether the rendered DOM matched the version a logged-in user would see.",
            ],
        )
        _write_evidence_sidecar(out, ev, log)
    else:
        if not probe_npx():
            print("Archive mode needs npx (apt install nodejs npm).")
            return EXIT_MISSING_DEPS
        out = output_dir / slugify(norm, "archive")
        cli_version = _cfg_str(config, "archive_cli_version",
                               DEFAULT_CONFIG["archive_cli_version"])
        archive_timeout = _cfg_int(config, "archive_timeout_s",
                                   DEFAULT_CONFIG["archive_timeout_s"], log)
        if not run_archive(norm, out, log, cli_version=cli_version,
                           timeout_s=archive_timeout):
            return EXIT_ARCHIVE_FAIL
        ev = evidence_wrap(
            claim=f"Captured self-contained HTML archive of {redact_url(norm)}",
            evidence=[{
                "artifact": str(out),
                "source_url": redact_url(norm),
                "bytes": out.stat().st_size,
                "tool": f"single-file-cli@{cli_version}",
            }],
            limitations=[
                "Assets served under auth/cookies or cross-origin policies may be missing.",
            ],
            unknowns=["Whether lazy-loaded media beyond the captured DOM was inlined."],
        )
        _write_evidence_sidecar(out, ev, log)
    print(f"Saved: {out}")
    print(f"Open:  {out.resolve().as_uri()}")
    print(f"Size:  {out.stat().st_size:,} bytes")
    print(f"Evidence: {_evidence_sidecar_path(out)}")
    return EXIT_OK


def cmd_fanout(query, output_dir, config, log):
    if not (query or "").strip():
        print("No query provided.")
        return EXIT_BAD_URL
    timeout = _cfg_int(config, "http_timeout", 20, log)
    ttl = _cfg_int(config, "fanout_cache_ttl_s",
                   DEFAULT_CONFIG["fanout_cache_ttl_s"], log)
    mmr_lambda = _cfg_float(config, "mmr_lambda", DEFAULT_CONFIG["mmr_lambda"], log)
    mmr_cap = _cfg_int(config, "mmr_cap", DEFAULT_CONFIG["mmr_cap"], log)
    _say(f"Fanning out across {len(FREE_API_PROVIDERS) + 1} free APIs for: {query!r} ...")
    t0 = time.time()
    hits = fan_out_search(query, timeout=timeout, log=log,
                          extra_token=os.environ.get("GITHUB_TOKEN"),
                          out_dir=output_dir, cache_ttl_s=ttl,
                          mmr_lambda=mmr_lambda, mmr_cap=mmr_cap)
    elapsed = time.time() - t0
    _say(f"Got {len(hits)} hits in {elapsed:.2f}s")
    by_src = {}
    for h in hits:
        by_src.setdefault(h["source"], []).append(h)
    for src, items in sorted(by_src.items()):
        _say(f"\n--- {src} ({len(items)}) ---")
        for it in items[:5]:
            _say(f"  * {it.get('title')}")
            if it.get("url"):
                _say(f"    {it['url']}")
    out = output_dir / f"fanout_{slugify('https://x/' + query, 'fanout', 'json')}"
    payload = evidence_wrap(
        claim=f"Search fan-out for query: {query}",
        evidence=hits,
        limitations=[
            "Free public APIs rate-limit; results may be truncated or stale.",
            "github-code search requires a token and is skipped otherwise.",
            "Ranking is heuristic (keyword overlap + per-source weight + MMR).",
        ],
        unknowns=[
            "Whether hits are currently relevant; freshness is not verified per-item.",
        ],
        extra={
            "query": query,
            "elapsed_s": round(elapsed, 3),
            "sources": sorted(by_src),
            "mmr_lambda": mmr_lambda,
            "mmr_cap": mmr_cap,
        },
    )
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nFull results: {out}")
    except OSError:
        pass
    log.info(f"fanout_saved hits={len(hits)}")
    return EXIT_OK


def cmd_ghidra(binary_path, keywords, script_dir, output_dir, log):
    if not (binary_path or "").strip():
        print("No binary path provided.")
        return EXIT_BAD_URL
    bp = Path(binary_path).expanduser().resolve()
    if not bp.is_file():
        print(f"Not a file: {bp}")
        return EXIT_BAD_URL
    java_ok, jv = probe_java()
    if not java_ok:
        print(f"Java 21+ required for Ghidra ({jv}). Install: sudo apt install openjdk-21-jdk")
        return EXIT_MISSING_DEPS
    gh = find_ghidra(script_dir)
    if not gh:
        cache = _load_ghidra_cache(script_dir)
        if cache.get("path"):
            _say(f"Cached Ghidra path was {cache['path']} but analyzeHeadless not found; "
                 f"will re-download.")
        ans = _ask("Ghidra not found. Download from GitHub releases (~400 MB)? [y/N]: ").lower()
        if ans != "y":
            print("Aborted.")
            return EXIT_MISSING_DEPS
        try:
            gh = download_ghidra(script_dir / "ghidra_dl", log, script_dir=script_dir)
        except Exception as e:
            print(f"Download failed: {e}")
            return EXIT_MISSING_DEPS
    kws = [k.strip() for k in (keywords or "").split(",") if k.strip()]
    print(f"Analyzing {bp} with Ghidra...")
    res = ghidra_headless_run(gh, bp, script_dir / "ghidra_proj",
                              script_dir, kws, log)
    out = output_dir / f"ghidra_{bp.stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    payload = evidence_wrap(
        claim=f"Ghidra headless analysis of {bp.name}",
        evidence=[res],
        limitations=[
            "Decompilation is not included; only function list, addresses, and "
            "printable ASCII strings are dumped.",
            "Analysis quality depends on Ghidra's auto-analysis and the timeout.",
        ],
        unknowns=[
            "Semantics of the recovered functions are not asserted; only raw evidence is recorded.",
        ],
        extra={"binary": str(bp), "keywords": kws},
    )
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"Result: {out}")
    except OSError as e:
        print(f"Write failed: {e}")
        return EXIT_WRITE_FAIL

    dump = res.get("dump") or {}
    n_funcs = len(dump.get("functions") or []) if isinstance(dump, dict) else 0
    n_strings = len(dump.get("strings") or []) if isinstance(dump, dict) else 0
    print(f"Functions: {n_funcs}  Strings: {n_strings}  "
          f"PostScriptUsed: {res.get('postscript_used')}")

    if res.get("keyword_hits"):
        for kw, lines in res["keyword_hits"].items():
            print(f"\n[{kw}] {len(lines)} string hit(s)")
            for ln in lines[:5]:
                print(f"  {ln}")
    return EXIT_OK


def cmd_generate(lang, idea, output_dir, log, search_hits=None, cfg=None,
                 run_generated=False):
    if lang not in GENERATORS:
        print(f"Unsupported lang: {lang}")
        return EXIT_BAD_URL
    if not (idea or "").strip():
        print("No idea provided.")
        return EXIT_BAD_URL
    _say(f"Generating {lang} skeleton (LLM if available, template otherwise)...")
    provider, code = synthesize_skeleton(lang, idea, search_hits, log, cfg=cfg)
    if provider is None or code is None:
        print("No LLM available and require_local_llm is set.")
        print("Run rea_mini.py interactively to install Ollama, "
              "or pass --no-require-local-llm to fall back to deterministic templates.")
        return EXIT_NO_LOCAL_LLM

    ext = GENERATORS[lang][1]
    out = output_dir / f"skel_{uuid.uuid4().hex[:8]}_" \
                       f"{slugify('https://x/' + idea[:60], 'gen', ext)}"
    try:
        atomic_write_text(out, code)
    except OSError as e:
        print(f"Write failed: {e}")
        return EXIT_WRITE_FAIL

    # Safe default: syntax check only. Full smoke test only under --run-generated.
    syntax = _syntax_check_skeleton(lang, out, log)
    smoke = None
    fallback_needed = bool(syntax.get("executed") and not syntax.get("passed"))

    if not fallback_needed and run_generated and provider not in ("template", "template_fallback"):
        smoke = _smoke_test_skeleton(lang, out, log)
        if smoke.get("executed") and not smoke.get("passed"):
            fallback_needed = True

    if fallback_needed and provider not in ("template", "template_fallback"):
        if log:
            log.warn(f"skeleton_quality_gate_failed provider={provider} lang={lang} "
                     f"syntax_passed={syntax.get('passed')} "
                     f"smoke_passed={None if smoke is None else smoke.get('passed')}")
        fallback_fn, _ = GENERATORS[lang]
        try:
            atomic_write_text(out, fallback_fn(idea))
        except OSError as e:
            print(f"Fallback write failed: {e}")
            return EXIT_WRITE_FAIL
        provider = "template_fallback"
        syntax = _syntax_check_skeleton(lang, out, log)
        smoke = _smoke_test_skeleton(lang, out, log) if run_generated else None

    quality_gate = {
        "syntax_check": syntax,
        "smoke_test": smoke,
        "ran_smoke_test": bool(run_generated),
        "initial_provider": provider,
        "fell_back_to_template": provider == "template_fallback",
    }

    ev = evidence_wrap(
        claim=f"Generated {lang} program that reconstructs: {idea[:100]}",
        evidence=[{
            "provider": provider,
            "artifact": str(out),
            "bytes": out.stat().st_size,
            "search_hits": (search_hits or [])[:30],
            "quality_gate": quality_gate,
        }],
        limitations=[
            "Generated code requires human review before deployment; templates are demos, "
            "not reconstructions.",
            "Quality gate only exercises syntax (and optionally the artifact's --test "
            "mode); it cannot prove correctness.",
        ],
        unknowns=[
            "Whether the generated logic matches the original target's behavior.",
        ],
    )
    ev_path = _write_evidence_sidecar(out, ev, log)
    print(f"Provider: {provider}")
    print(f"Saved:    {out}")
    print(f"Evidence: {ev_path}")
    print(f"Size:     {out.stat().st_size:,} bytes")
    print(f"Syntax:   executed={syntax.get('executed')} passed={syntax.get('passed')}")
    if smoke is not None:
        print(f"Smoke:    executed={smoke.get('executed')} passed={smoke.get('passed')}")
    else:
        print("Smoke:    skipped (pass --run-generated to execute the artifact's --test)")

    if provider in ("template", "template_fallback"):
        if provider == "template_fallback":
            print("(LLM output rejected by quality gate; used deterministic template.)")
        else:
            print("(No LLM available; used deterministic template because "
                  "--no-require-local-llm was set.)")
    return EXIT_OK


def cmd_full_pipeline(url, lang, output_dir, config, log, run_generated=False):
    norm = normalize_url(url)
    if not norm:
        print(f"Invalid URL: {url}")
        return EXIT_BAD_URL
    kw = " ".join(keywords_from(norm))
    if not kw:
        host = urllib.parse.urlparse(norm).netloc
        if host.startswith("www."):
            host = host[4:]
        kw = host.split(".")[0] if host else ""
    if not kw:
        print("Could not derive a search query from URL.")
        return EXIT_BAD_URL
    _say(f"[1/3] Deriving keywords: {kw!r}")
    _say("[2/3] Fan-out across free APIs...")
    timeout = _cfg_int(config, "http_timeout", 20, log)
    ttl = _cfg_int(config, "fanout_cache_ttl_s",
                   DEFAULT_CONFIG["fanout_cache_ttl_s"], log)
    mmr_lambda = _cfg_float(config, "mmr_lambda", DEFAULT_CONFIG["mmr_lambda"], log)
    mmr_cap = _cfg_int(config, "mmr_cap", DEFAULT_CONFIG["mmr_cap"], log)
    hits = fan_out_search(kw, timeout=timeout, log=log,
                          extra_token=os.environ.get("GITHUB_TOKEN"),
                          out_dir=output_dir, cache_ttl_s=ttl,
                          mmr_lambda=mmr_lambda, mmr_cap=mmr_cap)
    _say(f"      {len(hits)} hits")
    idea = f"Reimplement capability from {redact_url(norm)} (keywords: {kw})"
    _say(f"[3/3] Synthesizing {lang} skeleton...")
    return cmd_generate(lang, idea, output_dir, log, search_hits=hits, cfg=config,
                        run_generated=run_generated)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
async def _gather_selfcheck(script_dir: Path):
    report = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "checks": {},
        "optional_env": {},
    }
    checks = report["checks"]

    checks["playwright"] = "ok" if PLAYWRIGHT_OK else "MISSING"
    checks["readability-lxml"] = "ok" if READABILITY_OK else "MISSING"
    checks["html2text"] = "ok" if HTML2TEXT_OK else "MISSING"

    chromium_ok, chromium_info = False, "playwright_not_installed"
    if PLAYWRIGHT_OK:
        try:
            chromium_ok, chromium_info = await probe_chromium()
        except Exception as e:
            chromium_ok, chromium_info = False, f"probe_err={type(e).__name__}"
    checks["chromium"] = {"ok": chromium_ok, "detail": chromium_info}

    checks["node/npx"] = "ok" if probe_npx() else "MISSING"
    checks["strings"] = "ok" if probe_strings() else "MISSING"

    java_ok, jv = probe_java()
    checks["java21+"] = {"ok": java_ok, "detail": jv}

    gh = find_ghidra(script_dir)
    if gh:
        cache = _load_ghidra_cache(script_dir)
        if "supports_analysis_timeout" in cache:
            timeout_flag = bool(cache["supports_analysis_timeout"])
        else:
            timeout_flag = check_ghidra_timeout_flag(gh)
            cache["supports_analysis_timeout"] = timeout_flag
            _save_ghidra_cache(script_dir, cache)
        checks["ghidra"] = {"path": str(gh), "timeout_flag": timeout_flag}
    else:
        checks["ghidra"] = {"path": None, "note": "will download on first use"}

    # Lower Ollama probe timeout to 1s (matching the status line) so selfcheck
    # never blocks for 5s on an unreachable daemon.
    ollama_models = _ollama_list_models(timeout=1)
    if ollama_models:
        checks["ollama"] = f"ok ({len(ollama_models)} models)"
    else:
        checks["ollama"] = "optional"

    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
              "HF_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY",
              "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY"):
        report["optional_env"][k] = "ok" if os.environ.get(k) else "optional"

    return report


def _format_selfcheck_text(report: dict) -> str:
    lines = [f"python: {report['python']}", f"platform: {report['platform']}"]
    checks = report["checks"]
    for k in ("playwright", "readability-lxml", "html2text"):
        lines.append(f"  [{checks[k]}] {k}")
    c = checks["chromium"]
    lines.append(f"  [{'ok' if c['ok'] else 'MISSING'}] chromium ({c['detail']})")
    lines.append(f"  [{checks['node/npx']}] node/npx (archive mode)")
    lines.append(f"  [{checks['strings']}] strings (binutils)")
    j = checks["java21+"]
    lines.append(f"  [{'ok' if j['ok'] else 'MISSING'}] java 21+ for Ghidra ({j['detail']})")
    g = checks["ghidra"]
    if g.get("path"):
        lines.append(f"  [ok] ghidra ({g['path']})")
        lines.append(f"  [{'ok' if g.get('timeout_flag') else 'warn'}] "
                     f"ghidra -analysisTimeoutPerFile supported")
    else:
        lines.append("  [MISSING] ghidra (will download on first use)")
    lines.append(f"  [{checks['ollama']}] ollama local LLM")
    for k, v in report["optional_env"].items():
        lines.append(f"  [{v}] {k}")
    return "\n".join(lines)


async def cmd_selfcheck(script_dir, log, non_interactive=False, auto_install=False):
    print("Self-check:")
    report = await _gather_selfcheck(script_dir)
    print(_format_selfcheck_text(report))

    missing_pips = [p for p in ("playwright", "readability-lxml", "html2text")
                    if report["checks"][p] != "ok"]
    # `_missing_breakdown` already puts chromium first; no need to reshuffle.
    _, missing_nonpip = _missing_breakdown(report)

    should_install = False
    if missing_pips:
        if auto_install:
            should_install = True
            print(f"\n--yes set: auto-installing {' '.join(missing_pips)}")
        elif not non_interactive:
            ans = _ask(f"\nInstall missing pip packages ({' '.join(missing_pips)})? [Y/n]: ",
                       "y").lower()
            should_install = ans in ("y", "yes", "")
        # non_interactive without auto_install: skip install, report only.

    if should_install and missing_pips:
        if auto_install_pip(missing_pips, log):
            print(f"Installed: {' '.join(missing_pips)}")
            print("Re-detecting optional modules in-process...")
            _try_reimport_optional()
            report = await _gather_selfcheck(script_dir)
            print()
            print(_format_selfcheck_text(report))
            missing_pips = [p for p in missing_pips if report["checks"][p] != "ok"]

    if missing_pips:
        print("\nInstall missing pip packages (restart required if imports are cached):")
        print(f"  {sys.executable} -m pip install --user {' '.join(missing_pips)}")
    if missing_nonpip:
        print("\nInstall missing non-pip dependencies:")
        if "chromium" in missing_nonpip:
            print("  python3 -m playwright install chromium")
            print("  sudo apt install libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 "
                  "libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 "
                  "libgbm1 libpango-1.0-0 libcairo2 libasound2")
        if "node/npm" in missing_nonpip:
            print("  sudo apt install nodejs npm")
        if "binutils" in missing_nonpip:
            print("  sudo apt install binutils")
        if "openjdk-21" in missing_nonpip:
            print("  sudo apt install openjdk-21-jdk")

    if missing_nonpip or missing_pips:
        log.block("ERROR", ["selfcheck_missing"] + missing_nonpip +
                  ([f"pip:{p}" for p in missing_pips] if missing_pips else []))
        return EXIT_MISSING_DEPS

    print("\nAll checks passed.")
    log.info("selfcheck_ok")
    return EXIT_OK


# ---------------------------------------------------------------------------
# MCP stdio server (JSON-RPC 2.0, newline-delimited)
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_TOOL_TIMEOUT_S = 300


def _mcp_tools_schema():
    return [
        {
            "name": "rea_capture_markdown",
            "description": "Capture a URL and return clean LLM-friendly Markdown.",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
        {
            "name": "rea_fanout",
            "description": "Search free public APIs in parallel and return ranked hits.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "rea_ghidra_analyze",
            "description": "Run Ghidra headless on a local binary and return function "
                           "and string evidence as JSON.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "binary": {"type": "string"},
                    "keywords": {"type": "string", "description": "comma-separated"},
                },
                "required": ["binary"],
            },
        },
        {
            "name": "rea_generate_skeleton",
            "description": "Generate a runnable skeleton in HTML/JS/Python/Rust/C++ "
                           "for the given idea.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "idea": {"type": "string"},
                    "lang": {"type": "string", "enum": list(GENERATORS.keys())},
                },
                "required": ["idea"],
            },
        },
        {
            "name": "rea_full_pipeline",
            "description": "URL -> keyword derivation -> fan-out -> skeleton.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "lang": {"type": "string", "enum": list(GENERATORS.keys())},
                },
                "required": ["url"],
            },
        },
        {
            "name": "rea_selfcheck",
            "description": "Report installed dependencies and their status.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _mcp_text(s, is_error=False):
    return {"content": [{"type": "text", "text": s}], "isError": is_error}


async def _mcp_dispatch(name, args, script_dir, out_dir, config, log):
    if name == "rea_capture_markdown":
        url = args.get("url", "")
        norm = normalize_url(url)
        if not norm:
            return _mcp_text(f"Invalid URL: {url}", True)
        if not (READABILITY_OK and HTML2TEXT_OK):
            return _mcp_text("Missing deps: need readability-lxml + html2text", True)
        final_url, page_html = None, None
        if PLAYWRIGHT_OK:
            final_url, page_html = await _capture_with_retry(norm, config, log)
        if page_html is None:
            final_url, page_html = _urllib_capture(norm, config, log)
        if page_html is None:
            return _mcp_text("Navigation failed.", True)
        md = extract_for_llm(page_html, final_url, bool(config.get("ignore_images")), log)
        if md is None:
            return _mcp_text("Extraction returned empty.", True)
        return _mcp_text(md)

    if name == "rea_fanout":
        q = args.get("query", "")
        if not q.strip():
            return _mcp_text("Empty query.", True)
        timeout = _cfg_int(config, "http_timeout", 20, log)
        ttl = _cfg_int(config, "fanout_cache_ttl_s",
                       DEFAULT_CONFIG["fanout_cache_ttl_s"], log)
        mmr_lambda = _cfg_float(config, "mmr_lambda", DEFAULT_CONFIG["mmr_lambda"], log)
        mmr_cap = _cfg_int(config, "mmr_cap", DEFAULT_CONFIG["mmr_cap"], log)
        hits = fan_out_search(q, timeout=timeout, log=log,
                              extra_token=os.environ.get("GITHUB_TOKEN"),
                              out_dir=out_dir, cache_ttl_s=ttl,
                              mmr_lambda=mmr_lambda, mmr_cap=mmr_cap)
        payload = evidence_wrap(
            claim=f"Search fan-out for query: {q}",
            evidence=hits,
            limitations=["Free public APIs rate-limit; results may be truncated."],
            unknowns=["Relevance ranking is heuristic (keyword overlap + MMR)."],
            extra={"query": q, "mmr_lambda": mmr_lambda, "mmr_cap": mmr_cap},
        )
        return _mcp_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if name == "rea_ghidra_analyze":
        binary = args.get("binary", "")
        kws_str = args.get("keywords", "") or ""
        bp = Path(binary).expanduser().resolve()
        if not bp.is_file():
            return _mcp_text(f"Not a file: {bp}", True)
        java_ok, jv = probe_java()
        if not java_ok:
            return _mcp_text(f"Java 21+ required ({jv}).", True)
        gh = find_ghidra(script_dir)
        if not gh:
            return _mcp_text("Ghidra not found; run the CLI self-check first.", True)
        kws = [k.strip() for k in kws_str.split(",") if k.strip()]
        res = ghidra_headless_run(gh, bp, script_dir / "ghidra_proj",
                                  script_dir, kws, log)
        payload = evidence_wrap(
            claim=f"Ghidra headless analysis of {bp.name}",
            evidence=[res],
            limitations=["No decompilation; only functions + printable strings."],
            unknowns=["Semantics of recovered functions are not asserted."],
            extra={"binary": str(bp), "keywords": kws},
        )
        return _mcp_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    if name == "rea_generate_skeleton":
        idea = args.get("idea", "")
        lang = args.get("lang") or "python"
        if lang not in GENERATORS:
            return _mcp_text(f"Unsupported lang: {lang}", True)
        if not idea.strip():
            return _mcp_text("Empty idea.", True)
        provider, code = synthesize_skeleton(lang, idea, None, log, cfg=config)
        if provider is None or code is None:
            return _mcp_text(
                "No LLM available (require_local_llm is set). Configure Ollama "
                "or pass --no-require-local-llm.", True)
        return _mcp_text(f"provider: {provider}\n---\n{code}")

    if name == "rea_full_pipeline":
        url = args.get("url", "")
        lang = args.get("lang") or "python"
        norm = normalize_url(url)
        if not norm:
            return _mcp_text(f"Invalid URL: {url}", True)
        if lang not in GENERATORS:
            return _mcp_text(f"Unsupported lang: {lang}", True)
        kw = " ".join(keywords_from(norm))
        if not kw:
            host = urllib.parse.urlparse(norm).netloc
            if host.startswith("www."):
                host = host[4:]
            kw = host.split(".")[0] if host else ""
        if not kw:
            return _mcp_text("Could not derive keywords from URL.", True)
        timeout = _cfg_int(config, "http_timeout", 20, log)
        ttl = _cfg_int(config, "fanout_cache_ttl_s",
                       DEFAULT_CONFIG["fanout_cache_ttl_s"], log)
        mmr_lambda = _cfg_float(config, "mmr_lambda", DEFAULT_CONFIG["mmr_lambda"], log)
        mmr_cap = _cfg_int(config, "mmr_cap", DEFAULT_CONFIG["mmr_cap"], log)
        hits = fan_out_search(kw, timeout=timeout, log=log,
                              extra_token=os.environ.get("GITHUB_TOKEN"),
                              out_dir=out_dir, cache_ttl_s=ttl,
                              mmr_lambda=mmr_lambda, mmr_cap=mmr_cap)
        idea = f"Reimplement capability from {redact_url(norm)} (keywords: {kw})"
        provider, code = synthesize_skeleton(lang, idea, hits, log, cfg=config)
        if provider is None or code is None:
            return _mcp_text(
                "No LLM available (require_local_llm is set). Configure Ollama "
                "or pass --no-require-local-llm.", True)
        return _mcp_text(f"provider: {provider}\nkeywords: {kw}\n"
                         f"hits: {len(hits)}\n---\n{code}")

    if name == "rea_selfcheck":
        report = await _gather_selfcheck(script_dir)
        return _mcp_text(_format_selfcheck_text(report))

    return _mcp_text(f"Unknown tool: {name}", True)


async def _mcp_dispatch_with_timeout(name, args, script_dir, out_dir,
                                     config, log, timeout_s=MCP_TOOL_TIMEOUT_S):
    try:
        return await asyncio.wait_for(
            _mcp_dispatch(name, args, script_dir, out_dir, config, log),
            timeout=timeout_s)
    except asyncio.TimeoutError:
        if log:
            log.warn(f"mcp_tool_timeout name={name} timeout_s={timeout_s}")
        return _mcp_text(f"Tool {name} timed out after {timeout_s}s", True)
    except Exception as e:
        if log:
            log.error(f"mcp_tool_err name={name} err={type(e).__name__}: {e}")
        return _mcp_text(f"Tool {name} failed: {type(e).__name__}: {e}", True)


async def run_mcp_server(script_dir, out_dir, config, log):
    stdin = sys.stdin
    stdout = sys.stdout
    if stdin.isatty():
        print("MCP: reading from TTY — send JSON-RPC lines or press Ctrl+C to exit.",
              file=sys.stderr)
    log.info("mcp_server_start")
    while True:
        try:
            # Read stdin off the event loop so long-running tool calls don't
            # block the server from receiving the next request.
            line = await asyncio.to_thread(stdin.readline)
        except KeyboardInterrupt:
            log.info("mcp_server_stop user_interrupt")
            return EXIT_OK
        except Exception as e:
            log.warn(f"mcp_stdin_err err={type(e).__name__}")
            return EXIT_OK
        if not line:
            log.info("mcp_server_stop eof")
            return EXIT_OK
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
            continue

        method = msg.get("method")
        rid = msg.get("id")
        is_notification = "id" not in msg

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "rea_mini", "version": "1.2.0"},
                }
            elif method == "notifications/initialized":
                continue
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _mcp_tools_schema()}
            elif method == "tools/call":
                params = msg.get("params") or {}
                name = params.get("name")
                arguments = params.get("arguments") or {}
                result = await _mcp_dispatch_with_timeout(
                    name, arguments, script_dir, out_dir, config, log)
            else:
                if is_notification:
                    continue
                resp = {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32601,
                                  "message": f"method not found: {method}"}}
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()
                continue
        except Exception as e:
            log.error(f"mcp_dispatch_error err={type(e).__name__}: {e}")
            if is_notification:
                continue
            resp = {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"}}
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
            continue

        if is_notification:
            continue
        resp = {"jsonrpc": "2.0", "id": rid, "result": result}
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()


# ---------------------------------------------------------------------------
# Safe menu helper
# ---------------------------------------------------------------------------
async def _safe_call(fn, *args, log=None, **kwargs):
    try:
        r = fn(*args, **kwargs)
        if asyncio.iscoroutine(r):
            r = await r
        return r
    except KeyboardInterrupt:
        print("\n[interrupted, returning to menu]")
        return 130
    except Exception as e:
        if log:
            log.error(f"menu_action_failed err={type(e).__name__}: {e}")
        print(f"\nError: {e}")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(argv):
    global _QUIET
    log = None
    try:
        try:
            script_dir = Path(__file__).resolve().parent
        except NameError:
            script_dir = Path.cwd()
        cwd = Path.cwd()

        parser = argparse.ArgumentParser(prog="rea_mini.py", add_help=True)
        parser.add_argument("--cmd", choices=["capture", "archive", "fanout",
                                              "ghidra", "generate", "full",
                                              "selfcheck", "mcp",
                                              "validate-evidence"])
        parser.add_argument("--url")
        parser.add_argument("--mode", choices=["llm", "archive"])
        parser.add_argument("--query")
        parser.add_argument("--binary")
        parser.add_argument("--keywords")
        parser.add_argument("--lang", choices=list(GENERATORS.keys()))
        parser.add_argument("--idea")
        parser.add_argument("-o", "--output")
        parser.add_argument("--timeout", type=int)
        parser.add_argument("--config")
        parser.add_argument("--wait-selector", default=None,
                            help="CSS selector to wait for before extracting (capture)")
        parser.add_argument("--scroll", action="store_true",
                            help="scroll to bottom before extracting (capture)")
        parser.add_argument("--evidence-file",
                            help="Path to evidence JSON (validate-evidence command)")
        parser.add_argument("-y", "--yes", "--auto-install", dest="yes",
                            action="store_true",
                            help="Auto-install missing pip deps and pull models without prompting")
        parser.add_argument("-q", "--quiet", action="store_true",
                            help="Suppress progress banners (results still printed)")
        parser.add_argument("--list-models", action="store_true",
                            help="Show resolved LLM configuration")
        parser.add_argument("--run-generated", action="store_true",
                            help="Also run the generated artifact's own --test mode "
                                 "(executes LLM-generated code; off by default)")
        parser.add_argument("--no-require-local-llm", action="store_true",
                            help="Do not require a local Ollama model; fall back to "
                                 "cloud or deterministic templates")
        args = parser.parse_args(argv)

        _QUIET = bool(args.quiet)

        log = Log(script_dir / "rea_mini_error.log", DEFAULT_CONFIG["log_max_bytes"])
        log.info(f"run_start pid={os.getpid()} cwd={cwd} argv={argv}")
        if cwd != script_dir:
            _say(f"Note: running from {cwd}; artifacts go to "
                 f"{cwd / 'rea_output'} unless --output is set; logs go to {script_dir}")

        if sys.version_info < (3, 9):
            log.fatal(f"abort reason=python_version got={sys.version.split()[0]}")
            return EXIT_PYTHON_OLD
        log.info(f"platform os={platform.system()} arch={platform.machine()}")

        cfg_path = Path(args.config) if args.config else (script_dir / "rea_mini_config.json")
        config = load_config(cfg_path, log)

        # CLI overrides
        if args.timeout is not None:
            if args.timeout <= 0:
                print("--timeout must be > 0", file=sys.stderr)
                return EXIT_BAD_URL
            config["timeout"] = args.timeout
        if args.output:
            config["output_dir"] = args.output
        if args.wait_selector is not None:
            config["wait_selector"] = args.wait_selector
        if args.scroll:
            config["scroll_to_bottom"] = True
        if args.no_require_local_llm:
            config["require_local_llm"] = False

        out_dir = Path(config.get("output_dir", DEFAULT_CONFIG["output_dir"]))
        if not out_dir.is_absolute():
            out_dir = cwd / out_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Cannot create output dir {out_dir}: {e}", file=sys.stderr)
            return EXIT_WRITE_FAIL

        # --list-models short-circuits everything else.
        if args.list_models:
            return cmd_list_models(config, log)

        # Non-interactive dispatch: --cmd implies non-interactive.
        if args.cmd:
            if args.cmd == "validate-evidence":
                if not args.evidence_file:
                    print("--evidence-file required for validate-evidence")
                    return EXIT_BAD_URL
                return cmd_validate_evidence(args.evidence_file, log)
            if args.cmd == "capture":
                mode = args.mode or "llm"
                return await cmd_capture(args.url or "", mode, out_dir, config, log)
            if args.cmd == "archive":
                return await cmd_capture(args.url or "", "archive", out_dir, config, log)
            if args.cmd == "fanout":
                return cmd_fanout(args.query or "", out_dir, config, log)
            if args.cmd == "ghidra":
                return cmd_ghidra(args.binary or "", args.keywords or "",
                                  script_dir, out_dir, log)
            if args.cmd == "generate":
                return cmd_generate(args.lang or "python", args.idea or "",
                                    out_dir, log, cfg=config,
                                    run_generated=bool(args.run_generated))
            if args.cmd == "full":
                return cmd_full_pipeline(args.url or "", args.lang or "python",
                                         out_dir, config, log,
                                         run_generated=bool(args.run_generated))
            if args.cmd == "selfcheck":
                return await cmd_selfcheck(script_dir, log,
                                           non_interactive=True,
                                           auto_install=bool(args.yes))
            if args.cmd == "mcp":
                return await run_mcp_server(script_dir, out_dir, config, log)
            # Fallthrough safety (unreachable because argparse validates choices)
            print(f"Unknown command: {args.cmd}")
            return EXIT_BAD_URL

        # Interactive startup wizard (only for TTY, non --cmd runs).
        if sys.stdin.isatty():
            wizard_ok = await _startup_wizard(script_dir, out_dir, config, log, args)
            # Persist any wizard decisions so the next run is instant.
            save_config(cfg_path, config, log)
            if not wizard_ok:
                print("Startup aborted. Re-run interactively to complete onboarding, "
                      "or use --cmd for non-interactive use.", file=sys.stderr)
                return EXIT_NO_LOCAL_LLM

        # Interactive menu loop
        while True:
            _print_menu(_llm_status_line(config))
            choice = _ask("Choice: ")
            if choice in ("0", "q", "quit", "exit"):
                log.info("user_quit")
                return EXIT_OK
            if choice == "1":
                url = _ask("URL: ")
                await _safe_call(cmd_capture, url, "llm", out_dir, config, log, log=log)
            elif choice == "2":
                url = _ask("URL: ")
                await _safe_call(cmd_capture, url, "archive", out_dir, config, log, log=log)
            elif choice == "3":
                q = _ask("Search query: ")
                if q:
                    await _safe_call(cmd_fanout, q, out_dir, config, log, log=log)
            elif choice == "4":
                b = _ask("Binary path: ")
                k = _ask("Keywords (comma-separated, optional): ")
                if b:
                    await _safe_call(cmd_ghidra, b, k, script_dir, out_dir, log, log=log)
            elif choice == "5":
                lang = _ask(f"Language {list(GENERATORS)}: ", "python").lower()
                idea = _ask("Idea / spec: ")
                if idea:
                    await _safe_call(cmd_generate, lang, idea, out_dir, log,
                                     None, config, bool(args.run_generated), log=log)
            elif choice == "6":
                url = _ask("URL (repo/page): ")
                lang = _ask(f"Output language {list(GENERATORS)}: ", "python").lower()
                if url:
                    await _safe_call(cmd_full_pipeline, url, lang, out_dir, config,
                                     log, bool(args.run_generated), log=log)
            elif choice == "7":
                await _safe_call(cmd_selfcheck, script_dir, log,
                                 False, bool(args.yes), log=log)
            else:
                print("Unknown choice.")

    except (KeyboardInterrupt, EOFError):
        if log:
            log.warn("interrupted")
        return 130
    except Exception as e:
        if log:
            log.fatal(f"abort reason=unexpected err={type(e).__name__}: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        if log:
            log.info("run_end")
            log.close()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(async_main(sys.argv[1:])))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)