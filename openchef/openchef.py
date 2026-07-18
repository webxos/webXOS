#!/usr/bin/env python3
"""
HUMBOLDT-CHEF v2.1 – LACK‑Enhanced Edition (FULLY FIXED DROPDOWN + AGENT INSTRUCTIONS)

- Fixed: initial and retry model fetches force server refresh (?refresh=true)
- Added lightweight instructional comments for each major section to aid modular development
- All features: /abstract, /ralph, reflection, proactive questioning, J‑space, STACK, etc.

Run with --cli for CLI mode, or without for web UI.
"""

import os
import sys
import asyncio
import json
import time
import uuid
import logging
import signal
import re
import shlex
import argparse
import subprocess
import shutil
import tempfile
import urllib.parse
import datetime
import random
import hashlib
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator
from collections import defaultdict, deque
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

# ==============================================================================
# SECTION 1: DEPENDENCY MANAGEMENT
# PURPOSE: Check and import required and optional libraries. Gracefully warn
#          about missing optional packages but abort if core ones are missing.
# EXTENSION: Add new optional imports here with similar try/except patterns.
# ==============================================================================
try:
    import aiohttp
    from aiohttp import web, WSMsgType, WSCloseCode
except ImportError:
    print("ERROR: aiohttp required. Install: pip install aiohttp")
    sys.exit(1)

try:
    import aiosqlite
except ImportError:
    print("ERROR: aiosqlite required. Install: pip install aiosqlite")
    sys.exit(1)

try:
    from croniter import croniter
except ImportError:
    croniter = None
    print("WARNING: croniter not installed; cron jobs will run every hour (install with: pip install croniter)")

try:
    from pydantic import BaseModel, Field, ValidationError, ConfigDict
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    print("ERROR: pydantic and pydantic-settings required.")
    sys.exit(1)

# Optional
try:
    import yaml
except ImportError:
    yaml = None
    print("WARNING: PyYAML not installed; MAML skills may not load (install with: pip install PyYAML)")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    print("WARNING: BeautifulSoup not installed; Siphon research will be limited (install with: pip install beautifulsoup4)")

try:
    from git import Repo
except ImportError:
    Repo = None
    print("WARNING: GitPython not installed; code moderation commits disabled (install with: pip install GitPython)")

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    print("WARNING: chromadb not installed; dataset vector storage disabled (install with: pip install chromadb)")

try:
    import psutil
except ImportError:
    psutil = None
    print("WARNING: psutil not installed; real-time metrics disabled (install with: pip install psutil)")

try:
    import numpy as np
except ImportError:
    np = None
    print("WARNING: numpy not installed; cosine similarity disabled (install with: pip install numpy)")

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("WARNING: tiktoken not installed; token counting may be inaccurate (install with: pip install tiktoken)")

# ==============================================================================
# SECTION 2: CONFIGURATION (Pydantic Settings)
# PURPOSE: Centralizes all configurable parameters. Reads from .env and environment
#          variables with prefix OPENCHEF_. Modify to change runtime behaviour.
# EXTENSION: Add new fields with defaults; they become settable via env.
# ==============================================================================
class Settings(BaseSettings):
    # Core directories
    base_dir: Path = Field(default=Path(__file__).parent)
    skills_dir: Path = Field(default=Path("skills"))
    memory_dir: Path = Field(default=Path("agent_memories"))
    stack_dir: Path = Field(default=Path("stack_templates"))
    thread_repo_dir: Path = Field(default=Path("thread_repos"))
    research_dir: Path = Field(default=Path("research"))
    logs_dir: Path = Field(default=Path("logs"))
    workspace_dir: Path = Field(default=Path("workspace"))
    db_path: Path = Field(default=Path("factory.db"))
    long_messages_dir: Path = Field(default=Path("long_messages"))
    error_log_path: Path = Field(default=Path("logs/error_log.md"))
    datasets_dir: Path = Field(default=Path("datasets"))
    lineage_dir: Path = Field(default=Path("lineage"))
    backups_dir: Path = Field(default=Path("backups"))

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_generate_timeout: int = 9999
    ollama_embed_timeout: int = 30
    ollama_short_timeout: int = 5
    default_generate_model: str = "qwen2.5:0.5b"
    default_embed_model: str = "nomic-embed-text"

    # Web Search API Keys
    serpapi_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    enable_duckduckgo: bool = True
    enable_serpapi: bool = False
    enable_firecrawl: bool = False

    # Agents
    max_iterations: int = 12
    token_budget: int = 8192
    sandbox_root: Path = Field(default=Path("workspace"))
    system_dirs: List[str] = ["skills", "stack_templates", "thread_repos", "research", "logs", "agent_memories", "datasets", "lineage", "backups"]
    forbidden_patterns: List[str] = ["/etc", "/root", ".ssh", "/proc", "/sys"]

    # Server
    host: str = "0.0.0.0"
    port: int = 3721
    secret_key: str = "CHANGE_ME_IN_PRODUCTION"

    # Model Clients (for future cloud providers)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    default_provider: str = "ollama"

    # Rate limiting
    rate_limit: int = 10
    rate_window: float = 1.0

    # Logging
    log_level: str = "INFO"

    # OpenTeamFormat (optional)
    team_dir: Path = Field(default=Path("open_team"))
    agents_yaml: Path = Field(default=Path("open_team/agent.yaml"))
    agents_md: Path = Field(default=Path("open_team/AGENTS.md"))
    team_skills_dir: Path = Field(default=Path("open_team/.agents/skills"))

    # Bio files
    bio_file: Path = Field(default=Path("bio.md"))
    soul_file: Path = Field(default=Path("soul.md"))

    # MAML
    maml_enabled: bool = True
    enable_auto_siphon: bool = True

    # Ralph
    max_generations: int = 30
    convergence_threshold: float = 0.95
    stagnation_limit: int = 3
    population_size: int = 3

    # Metrics broadcasting interval (seconds)
    metrics_broadcast_interval: int = 10

    # Backup interval (hours)
    backup_interval_hours: int = 24

    # WebSocket enhancements
    websocket_ping_interval: int = 15
    websocket_ping_timeout: int = 20
    max_message_size: int = 16 * 1024 * 1024  # 16MB

    # Skill execution limits
    skill_timeout: int = 30
    skill_max_output: int = 10000
    skill_max_memory_mb: int = 512
    skill_max_cpu_seconds: int = 10

    # Proactive questioning
    proactive_question_interval: int = 30  # seconds

    # J‑space context size
    workspace_context_size: int = 5  # number of concepts

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPENCHEF_")

settings = Settings()

# Create necessary directories
for d in [settings.skills_dir, settings.memory_dir, settings.stack_dir,
          settings.thread_repo_dir, settings.research_dir, settings.logs_dir,
          settings.workspace_dir, settings.long_messages_dir,
          settings.datasets_dir, settings.lineage_dir, settings.backups_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Error log
settings.error_log_path.parent.mkdir(parents=True, exist_ok=True)
if not settings.error_log_path.exists():
    settings.error_log_path.write_text("# Error Log\n\nNo errors logged yet.\n", encoding='utf-8')

# ==============================================================================
# SECTION 3: LOGGING (with rotation)
# PURPOSE: Set up structured logging with color output for terminals and file
#          rotation. Use logger.info/warning/error throughout.
# EXTENSION: Add custom log levels or additional handlers (e.g., syslog).
# ==============================================================================
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            settings.logs_dir / "openchef.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger("openchef")

COLORS = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
}

def colorize(text, color):
    if sys.stdout.isatty():
        return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"
    return text

# ==============================================================================
# SECTION 4: HELPERS (safe_path, file ops, async subprocess, JSON, etc.)
# PURPOSE: Reusable utilities for path validation, file I/O, subprocess,
#          JSON extraction, and other common tasks. These are used throughout.
# EXTENSION: Add new helper functions here (e.g., for data validation).
# ==============================================================================
def safe_path(path: Union[str, Path]) -> Path:
    """Safely resolve a path, ensuring it stays within allowed directories."""
    if not path:
        raise ValueError("Empty path provided")
    abs_path = Path(path).resolve()
    base = Path(settings.base_dir).resolve()
    sandbox = Path(settings.sandbox_root).resolve()
    allowed_roots = [base, sandbox] + [Path(d).resolve() for d in settings.system_dirs]
    for root in allowed_roots:
        try:
            if abs_path.is_relative_to(root):
                return abs_path
        except AttributeError:
            if str(abs_path).startswith(str(root) + os.sep) or str(abs_path) == str(root):
                return abs_path
    raise PermissionError(f"Path {abs_path} is not allowed.")

def read_file(path: Union[str, Path]) -> str:
    try:
        return safe_path(path).read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"read_file {path}: {e}")
        return ""

def write_file(path: Union[str, Path], content: str) -> None:
    try:
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
    except Exception as e:
        logger.error(f"write_file {path}: {e}")

def append_to_file(path: Union[str, Path], content: str) -> None:
    try:
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('a', encoding='utf-8') as f:
            f.write(content + "\n")
    except Exception as e:
        logger.error(f"append_to_file {path}: {e}")

def log_error_to_file(error_msg: str):
    ts = datetime.datetime.now().isoformat()
    entry = f"\n## {ts}\n{error_msg}\n"
    append_to_file(settings.error_log_path, entry)

def _set_child_limits(memory_limit_mb: Optional[int], cpu_limit_seconds: Optional[int]):
    """Set resource limits for a child process. Called via preexec_fn."""
    if sys.platform.startswith('linux'):
        try:
            import resource
            if memory_limit_mb:
                mem = memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            if cpu_limit_seconds:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds + 1))
        except Exception as e:
            logger.warning(f"Could not set child process limits: {e}")

async def run_cmd_async(cmd: List[str], cwd: Optional[Union[str, Path]] = None,
                         timeout: int = 30, env: Optional[Dict] = None,
                         memory_limit_mb: Optional[int] = None,
                         cpu_limit_seconds: Optional[int] = None) -> Tuple[str, str, int]:
    """Non‑blocking subprocess with resource limits in child (Unix only)."""
    if cwd:
        cwd = safe_path(cwd)
    try:
        preexec_fn = None
        if sys.platform.startswith('linux'):
            preexec_fn = lambda: _set_child_limits(memory_limit_mb, cpu_limit_seconds)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            preexec_fn=preexec_fn
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return (stdout.decode('utf-8', errors='ignore'),
                    stderr.decode('utf-8', errors='ignore'),
                    proc.returncode)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "", f"Command timed out after {timeout}s", -1
    except FileNotFoundError as e:
        return "", str(e), -1
    except Exception as e:
        log_error_to_file(f"run_cmd_async {cmd}: {e}")
        return "", str(e), -1

def extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    cleaned = re.sub(r'```(?:json)?\s*', '', text, flags=re.I)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.I)
    cleaned = re.sub(r'(\{|\,)\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', cleaned)
    cleaned = re.sub(r',\s*(\}|\])', r'\1', cleaned)
    cleaned = cleaned.replace("'", '"')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r'(\{[\s\S]*?\})', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None

def repair_json(raw: str) -> str:
    if not raw:
        return raw
    raw = re.sub(r'```(?:json)?\s*', '', raw, flags=re.I)
    raw = re.sub(r'```\s*$', '', raw, flags=re.I)
    raw = re.sub(r'(\{|\,)\s*([a-zA-Z0-9_]+)\s*\:', r'\1"\2":', raw)
    raw = re.sub(r',\s*\}', '}', raw)
    raw = re.sub(r',\s*\]', ']', raw)
    raw = raw.replace("'", '"')
    open_braces = raw.count('{')
    close_braces = raw.count('}')
    open_brackets = raw.count('[')
    close_brackets = raw.count(']')
    if open_braces > close_braces:
        raw += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        raw += ']' * (open_brackets - close_brackets)
    return raw

def ensure_code_block(text: str, language: str = 'auto') -> str:
    if '```' in text:
        return text
    if re.search(r'<\s*html|def\s+|function\s*\(|class\s+|import\s+|require\s*\(', text, re.I):
        return f"```{language}\n{text}\n```"
    return text

def cosine_sim(a: List[float], b: List[float]) -> float:
    if not np:
        return 0.0
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_channel_personality(channel_name: str) -> Dict:
    if channel_name == 'random':
        return {'temperature': 1.2, 'system_bonus': "\nYou are in #random. Be creative, humorous."}
    elif channel_name == 'siphon':
        return {'temperature': 0.2, 'system_bonus': "\nYou are in #siphon. Be concise, factual."}
    elif channel_name == 'code':
        return {'temperature': 0.3, 'system_bonus': "\nSTRICT CODE CHANNEL: Output only code blocks."}
    else:
        return {'temperature': 0.7, 'system_bonus': ""}

# ==============================================================================
# SECTION 5: ASYNC TTL CACHE (in‑memory)
# PURPOSE: Simple key‑value store with time‑to‑live for caching short‑lived data.
# EXTENSION: Could be replaced with Redis or other distributed cache if needed.
# ==============================================================================
class TTLCache:
    """Simple in-memory async cache with TTL."""
    def __init__(self, default_ttl: int = 30):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.default_ttl = default_ttl

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry and entry['expires'] > time.time():
                return entry['value']
            else:
                if entry:
                    del self._cache[key]
                return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if ttl is None:
            ttl = self.default_ttl
        async with self._lock:
            self._cache[key] = {'value': value, 'expires': time.time() + ttl}

    async def delete(self, key: str):
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self):
        async with self._lock:
            self._cache.clear()

cache = TTLCache(default_ttl=30)

# ==============================================================================
# SECTION 6: OLLAMA CLIENT (with circuit breaker)
# PURPOSE: Async wrapper for Ollama's generate and embeddings endpoints.
#          Includes retries, circuit breaker, and token tracking.
# EXTENSION: Add support for other providers (OpenAI, Anthropic) by creating
#            a common interface and a factory.
# ==============================================================================
class OllamaCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0
        self.state = "CLOSED"
        self._lock = asyncio.Lock()

    async def call(self, coro):
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker: HALF_OPEN, testing...")
                else:
                    raise Exception("Ollama circuit open")
        try:
            result = await coro
            async with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker: CLOSED (recovered)")
            return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker: OPEN (failures={self.failure_count})")
            raise e

class OllamaClient:
    def __init__(self):
        self.circuit = OllamaCircuitBreaker()
        self._semaphore = asyncio.Semaphore(8)
        self.token_count = 0
        self.last_token_time = time.time()
        self._token_lock = asyncio.Lock()

    async def query(self, model: str, prompt: str, system_prompt: str = '',
                    temperature: float = 0.7, agent_id: Optional[str] = None,
                    degraded: bool = False) -> str:
        async with self._semaphore:
            return await self.circuit.call(
                self._query_ollama(model, prompt, system_prompt, temperature, agent_id, degraded)
            )

    async def _query_ollama(self, model, prompt, system, temp, agent_id, degraded):
        num_predict = 512 if degraded or re.search(r'0\.5b|1b', model, re.I) else 2048
        timeout_seconds = min(30, settings.ollama_generate_timeout) if degraded else settings.ollama_generate_timeout
        payload = {
            'model': model,
            'prompt': prompt[:4096],
            'system': system[:2048],
            'stream': False,
            'options': {
                'temperature': temp,
                'num_predict': num_predict,
                'num_ctx': 4096
            }
        }
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{settings.ollama_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            reply = data.get('response', '')
                            if 'eval_count' in data:
                                async with self._token_lock:
                                    self.token_count += data['eval_count']
                                    self.last_token_time = time.time()
                            return reply or "[OLLAMA_ERROR] Empty response"
                        else:
                            error_text = await resp.text()
                            logger.error(f"Ollama HTTP {resp.status}: {error_text[:200]}")
                            if resp.status == 500 and 'out of memory' in error_text.lower():
                                raise Exception("OOM")
                            backoff = min(30, 2 ** attempt)
                            await asyncio.sleep(backoff)
            except Exception as e:
                logger.error(f"Ollama attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        return "[OLLAMA_ERROR] All retries failed"

    async def get_embedding(self, text: str) -> List[float]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{settings.ollama_url}/api/embeddings",
                    json={"model": settings.default_embed_model, "prompt": text[:2000]},
                    timeout=settings.ollama_embed_timeout
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("embedding", [])
                    else:
                        logger.warning(f"Embedding error: {await resp.text()}")
                        return self._dummy_embedding()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return self._dummy_embedding()

    def _dummy_embedding(self) -> List[float]:
        return [random.random() for _ in range(384)]

ollama_client = OllamaClient()

# ==============================================================================
# SECTION 7: ASYNC DATABASE LAYER (aiosqlite)
# PURPOSE: Wraps SQLite with async methods and connection management.
#          Contains all table definitions and basic CRUD operations.
# EXTENSION: Add new tables here; all queries go through this class.
# ==============================================================================
class Database:
    def __init__(self):
        self._conn = None
        self._lock = asyncio.Lock()

    async def init(self):
        self._conn = await aiosqlite.connect(
            str(settings.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False
        )
        await self._conn.execute('PRAGMA journal_mode=WAL')
        await self._conn.execute('PRAGMA synchronous=NORMAL')
        await self._conn.execute('PRAGMA busy_timeout=30000')
        await self._conn.execute('PRAGMA cache_size=-20000')
        await self._create_tables()

    async def _create_tables(self):
        # messages table
        async with self._conn.execute('''CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            store_id TEXT,
            sender TEXT,
            sender_type TEXT,
            content TEXT,
            parent_id TEXT,
            thread_id TEXT,
            timestamp INTEGER,
            reactions TEXT,
            pinned INTEGER DEFAULT 0
        )'''):
            pass
        await self._conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_store_id ON messages(store_id)')
        await self._conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
        await self._conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id)')

        # agents table
        await self._conn.execute('''CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT,
            model TEXT,
            system_prompt TEXT,
            channels TEXT,
            strict_channel TEXT,
            status TEXT,
            is_embed_operator INTEGER DEFAULT 0,
            is_code_moderator INTEGER DEFAULT 0,
            last_response_time TEXT
        )''')

        # agent_memory table
        await self._conn.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
            agent_id TEXT PRIMARY KEY,
            e_pool TEXT,
            x_pool TEXT,
            weights TEXT,
            stats TEXT,
            last_update INTEGER
        )''')

        # project_states
        await self._conn.execute('''CREATE TABLE IF NOT EXISTS project_states (
            store_id TEXT PRIMARY KEY,
            state TEXT,
            timestamp INTEGER
        )''')

        # webhooks
        await self._conn.execute('''CREATE TABLE IF NOT EXISTS webhooks (
            name TEXT PRIMARY KEY,
            url TEXT,
            method TEXT DEFAULT 'POST',
            headers TEXT,
            created_at INTEGER
        )''')

        # cron_jobs
        await self._conn.execute('''CREATE TABLE IF NOT EXISTS cron_jobs (
            name TEXT PRIMARY KEY,
            schedule TEXT,
            webhook_name TEXT,
            enabled INTEGER DEFAULT 1,
            last_run INTEGER,
            next_run INTEGER,
            created_at INTEGER,
            FOREIGN KEY (webhook_name) REFERENCES webhooks(name)
        )''')

        # dataset_jobs
        await self._conn.execute('''CREATE TABLE IF NOT EXISTS dataset_jobs (
            id TEXT PRIMARY KEY,
            dataset_name TEXT,
            status TEXT,
            total_targets INTEGER,
            processed INTEGER,
            errors INTEGER,
            created_at INTEGER,
            updated_at INTEGER,
            result_summary TEXT
        )''')

        # stack_templates
        await self._conn.execute('''CREATE TABLE IF NOT EXISTS stack_templates (
            name TEXT PRIMARY KEY,
            content TEXT,
            description TEXT,
            tags TEXT,
            created_at INTEGER
        )''')
        await self._conn.commit()

    async def execute(self, query: str, params: tuple = ()):
        async with self._lock:
            async with self._conn.execute(query, params) as cursor:
                await self._conn.commit()
                return cursor

    async def fetchall(self, query: str, params: tuple = ()):
        async with self._lock:
            async with self._conn.execute(query, params) as cursor:
                return await cursor.fetchall()

    async def fetchone(self, query: str, params: tuple = ()):
        async with self._lock:
            async with self._conn.execute(query, params) as cursor:
                return await cursor.fetchone()

    async def close(self):
        if self._conn:
            await self._conn.close()

db = Database()

# ==============================================================================
# SECTION 8: BIO MANAGER
# PURPOSE: Manages the bio.md and soul.md files that provide agent context and
#          persistent memory. Updates heartbeat summaries.
# EXTENSION: Could be extended to support more markdown sections.
# ==============================================================================
class BioManager:
    def __init__(self):
        self.bio_path = settings.bio_file
        self.soul_path = settings.soul_file
        self._ensure_bio_exists()

    def _ensure_bio_exists(self):
        if not self.bio_path.exists():
            template = """# BIO.MD – Living Central Kitchen Agent
**Last Updated:** {timestamp}
**Role:** Coordinator of collaborative agents (cooks in a kitchen)

## SOUL
- Local-only, sandboxed Ollama agent.
- Helpful, concise, self-improving.
- Treat every user request as a refresher.

## SKILLS
MAML-powered tools (auto-loaded).

## MEMORY
{summary}

## CONTEXT
Working dir: {wd}
"""
            write_file(self.bio_path, template.format(
                timestamp=datetime.datetime.now().isoformat(),
                summary="Initial bootstrap",
                wd=os.getcwd()
            ))
        if not self.soul_path.exists():
            write_file(self.soul_path, "# SOUL.MD (legacy)\nSame as bio.md's SOUL section.\n")

    def read_bio(self) -> str:
        return read_file(self.bio_path)

    def read_soul(self) -> str:
        return read_file(self.soul_path)

    def update_timestamp(self):
        content = read_file(self.bio_path)
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("**Last Updated:**"):
                lines[i] = f"**Last Updated:** {datetime.datetime.now().isoformat()}"
                break
        write_file(self.bio_path, "\n".join(lines))

    async def heartbeat(self, user_message: str, agent_response: str):
        try:
            content = read_file(self.bio_path)
            ts = datetime.datetime.now().isoformat()
            summary_line = f"- {ts} | User: {user_message[:150]}... | Response: {agent_response[:100]}..."
            if "## MEMORY" in content:
                parts = content.split("## MEMORY", 1)
                before = parts[0]
                after = parts[1]
                next_section = re.search(r'\n## [A-Z]', after)
                if next_section:
                    insert_pos = next_section.start()
                    new_memory = after[:insert_pos] + f"\n{summary_line}\n" + after[insert_pos:]
                else:
                    new_memory = after + f"\n{summary_line}\n"
                new_content = before + "## MEMORY" + new_memory
            else:
                new_content = content + f"\n## MEMORY\n{summary_line}\n"
            new_content = re.sub(r'\*\*Last Updated:\*\* .*', f'**Last Updated:** {ts}', new_content)
            write_file(self.bio_path, new_content)
        except Exception as e:
            logger.warning(f"Bio heartbeat failed: {e}")

bio = BioManager()

# ==============================================================================
# SECTION 9: MAML SKILL ENGINE
# PURPOSE: Loads and executes skills from the skills directory. Supports
#          Python, Bash, C, and MAML (markdown with YAML frontmatter) skills.
# EXTENSION: Add new skill types (e.g., JavaScript) by extending _run_skill and
#            load methods.
# ==============================================================================
class SkillMetadata(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str
    triggers: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    args: Optional[Dict[str, Any]] = None
    examples: List[str] = Field(default_factory=list)
    tests: List[Dict] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    timeout: int = 30
    sandbox_level: str = "low"
    language: str = "python"

class SkillEngine:
    def __init__(self):
        self.registry: Dict[str, Dict] = {}
        self._loaded = False

    async def reload(self):
        self.registry.clear()
        for file in settings.skills_dir.glob("*.maml.md"):
            await self._load_skill_maml(file)
        for file in settings.skills_dir.glob("*.py"):
            if not file.name.endswith(".maml.md"):
                self._load_skill_py(file)
        for file in settings.skills_dir.glob("*.sh"):
            self._load_skill_sh(file)
        for file in settings.skills_dir.glob("*.c"):
            self._load_skill_c(file)
        self._loaded = True
        logger.info(f"Loaded {len(self.registry)} skills")
        return self.registry

    async def _load_skill_maml(self, path: Path):
        content = read_file(path)
        match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not match or not yaml:
            return
        try:
            data = yaml.safe_load(match.group(1))
            meta = SkillMetadata(**data)
        except Exception as e:
            logger.warning(f"Failed to parse skill metadata in {path}: {e}")
            return
        body = content.split('---', 2)[-1] if '---' in content else content
        self.registry[meta.name] = {
            "metadata": meta,
            "body": body,
            "path": path,
            "type": "maml",
            "usage": 0
        }

    def _load_skill_py(self, path: Path):
        content = read_file(path)
        name = desc = args = None
        for line in content.split('\n'):
            if line.startswith('# NAME:'):
                name = line.replace('# NAME:', '').strip()
            elif line.startswith('# DESCRIPTION:'):
                desc = line.replace('# DESCRIPTION:', '').strip()
            elif line.startswith('# ARGS:'):
                args = line.replace('# ARGS:', '').strip()
        if not name:
            name = path.stem
        if not desc:
            desc = "Auto-loaded Python skill"
        meta = SkillMetadata(name=name, description=desc, args={'args': args} if args else None)
        self.registry[name] = {
            "metadata": meta,
            "script": str(path),
            "type": "py",
            "usage": 0
        }

    def _load_skill_sh(self, path: Path):
        content = read_file(path)
        name = desc = args = None
        for line in content.split('\n'):
            if line.startswith('# NAME:'):
                name = line.replace('# NAME:', '').strip()
            elif line.startswith('# DESCRIPTION:'):
                desc = line.replace('# DESCRIPTION:', '').strip()
            elif line.startswith('# ARGS:'):
                args = line.replace('# ARGS:', '').strip()
        if not name:
            name = path.stem
        if not desc:
            desc = "Auto-loaded Bash skill"
        meta = SkillMetadata(name=name, description=desc, args={'args': args} if args else None)
        self.registry[name] = {
            "metadata": meta,
            "script": str(path),
            "type": "sh",
            "usage": 0
        }

    def _load_skill_c(self, path: Path):
        content = read_file(path)
        name = desc = args = None
        for line in content.split('\n'):
            if line.startswith('// NAME:'):
                name = line.replace('// NAME:', '').strip()
            elif line.startswith('// DESCRIPTION:'):
                desc = line.replace('// DESCRIPTION:', '').strip()
            elif line.startswith('// ARGS:'):
                args = line.replace('// ARGS:', '').strip()
        if not name:
            name = path.stem
        if not desc:
            desc = "Auto-loaded C skill"
        meta = SkillMetadata(name=name, description=desc, args={'args': args} if args else None)
        self.registry[name] = {
            "metadata": meta,
            "script": str(path),
            "type": "c",
            "usage": 0
        }

    def list(self) -> List[Dict]:
        return [{"name": k, "description": v["metadata"].description,
                 "capabilities": v["metadata"].capabilities,
                 "usage": v["usage"]} for k, v in self.registry.items()]

    async def execute(self, name: str, args: List[str], agent_id: Optional[str] = None) -> Dict:
        if name not in self.registry:
            return {"error": f"Skill {name} not found"}
        skill = self.registry[name]
        meta = skill["metadata"]
        safe_args = [re.sub(r'[^a-zA-Z0-9\-_.\/ ]', '', str(a)) for a in args]
        sandbox = tempfile.mkdtemp(prefix="skill_", dir=str(settings.workspace_dir))
        try:
            result = await self._run_skill(skill, safe_args, sandbox)
            return result
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    async def _run_skill(self, skill: Dict, args: List[str], cwd: Union[str, Path]) -> Dict:
        meta = skill["metadata"]
        try:
            if skill["type"] == "sh":
                cmd = ["bash", skill["script"]] + args
            elif skill["type"] == "py":
                cmd = ["python3", skill["script"]] + args
            elif skill["type"] == "c":
                src = Path(skill["script"])
                bin_path = src.with_suffix(".bin")
                if not bin_path.exists() or src.stat().st_mtime > bin_path.stat().st_mtime:
                    compile_cmd = ["gcc", str(src), "-o", str(bin_path)]
                    out, err, code = await run_cmd_async(compile_cmd, cwd=cwd, timeout=30)
                    if code != 0:
                        return {"error": f"Compilation failed: {err[:200]}"}
                cmd = [str(bin_path)] + args
            elif skill["type"] == "maml":
                return {"result": f"Skill {meta.name} is a markdown definition:\n{skill['body'][:500]}"}
            else:
                return {"error": f"Unknown skill type: {skill['type']}"}

            out, err, code = await run_cmd_async(
                cmd, cwd=cwd, timeout=meta.timeout,
                memory_limit_mb=settings.skill_max_memory_mb,
                cpu_limit_seconds=settings.skill_max_cpu_seconds
            )
            skill["usage"] += 1
            output = out.strip() or err.strip()
            if not output:
                return {"error": "No output from skill"}
            if len(output) > settings.skill_max_output:
                output = output[:settings.skill_max_output] + "... [truncated]"
            parsed = extract_json(output)
            return parsed if parsed else {"result": output, "warning": "Not JSON"}
        except asyncio.TimeoutError:
            return {"error": f"Skill {meta.name} timed out after {meta.timeout}s"}
        except Exception as e:
            log_error_to_file(f"Skill execution error: {e}")
            return {"error": str(e)[:200]}

    def find_by_capability(self, capability: str) -> List[Dict]:
        results = []
        for name, skill in self.registry.items():
            if capability in skill["metadata"].capabilities:
                results.append({"name": name, "metadata": skill["metadata"]})
        return results

    def search(self, query: str, k: int = 5) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for name, skill in self.registry.items():
            desc = skill["metadata"].description.lower()
            if query_lower in desc or any(q in desc for q in query_lower.split()):
                results.append({"name": name, "metadata": skill["metadata"]})
        return results[:k]

    async def reverse_skill(self, description: str, agent_id: Optional[str] = None) -> Dict:
        prompt = f"""Create a Python skill script based on this description:

{description}

Return ONLY valid Python code with:
1. A # NAME: comment at the top with the skill name
2. A # DESCRIPTION: comment
3. A # ARGS: comment (optional)
4. The main logic with proper error handling
5. Output as JSON: print(json.dumps({{"result": "..."}}))

Do not include any markdown, explanations, or extra text.
"""
        response = await ollama_client.query(
            settings.default_generate_model, prompt,
            temperature=0.3, agent_id=agent_id
        )
        if response.startswith('[OLLAMA_ERROR]'):
            return {"error": response}
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        name_match = re.search(r'# NAME:\s*(\w+)', code)
        if name_match:
            name = name_match.group(1)
            file_path = settings.skills_dir / f"{name}.py"
            write_file(file_path, code)
            file_path.chmod(0o755)
            await self.reload()
            return {"result": f"Skill {name} created successfully", "path": str(file_path)}
        return {"error": "Could not determine skill name from generated code"}

# ==============================================================================
# SECTION 10: DECENTMEM (Agent Memory)
# PURPOSE: Stores and retrieves agent experiences (e‑pool for high‑scoring,
#          x‑pool for low‑scoring) with embeddings for similarity retrieval.
# EXTENSION: Add more sophisticated memory retrieval (e.g., RAG over vectors).
# ==============================================================================
class DecentMem:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.e_pool = []
        self.x_pool = []
        self.weights = {"exploit": 0.6, "explore": 0.4}
        self.stats = {"judgements": 0, "avg": 50}
        self.lock = asyncio.Lock()
        self.load()

    def load(self):
        p = settings.memory_dir / f"{self.agent_id}.json"
        if p.exists():
            try:
                with p.open('r') as f:
                    d = json.load(f)
                    self.e_pool = d.get('e_pool', [])
                    self.x_pool = d.get('x_pool', [])
                    self.weights = d.get('weights', {"exploit": 0.6, "explore": 0.4})
                    self.stats = d.get('stats', {"judgements": 0, "avg": 50})
            except Exception as e:
                logger.warning(f"Memory load error for {self.agent_id}: {e}")

    def save(self):
        p = settings.memory_dir / f"{self.agent_id}.json"
        try:
            with p.open('w') as f:
                json.dump({
                    'e_pool': self.e_pool,
                    'x_pool': self.x_pool,
                    'weights': self.weights,
                    'stats': self.stats
                }, f)
        except Exception as e:
            logger.error(f"Memory save error: {e}")

    async def add(self, trajectory: str, score: int, task: str, embedding: Optional[List[float]] = None):
        async with self.lock:
            entry = {'trajectory': trajectory, 'score': score, 'task': task,
                     'ts': time.time(), 'embedding': embedding}
            if score >= 60:
                self.e_pool.insert(0, entry)
                if len(self.e_pool) > 50:
                    self.e_pool.pop()
            else:
                self.x_pool.insert(0, {'candidate': trajectory, 'score': score,
                                       'ts': time.time(), 'embedding': embedding})
                if len(self.x_pool) > 30:
                    self.x_pool.pop()
            self.stats['judgements'] += 1
            self.stats['avg'] = (self.stats['avg'] * (self.stats['judgements']-1) + score) / self.stats['judgements']
            self.save()

    async def judge(self, trajectory: str, context: str, model: str = 'qwen2.5:0.5b') -> int:
        prompt = f"Rate (0-100) this response to the query. Be generous with small models:\nQuery: {context[:300]}\nResponse: {trajectory[:500]}\nJSON: {{'score': number}}"
        try:
            response = await ollama_client.query(model, prompt, temperature=0.2)
            parsed = extract_json(response)
            if parsed and 'score' in parsed:
                score = min(100, max(0, int(parsed['score'])))
                emb = await ollama_client.get_embedding(trajectory)
                await self.add(trajectory, score, context, emb)
                delta = (score - 50) / 200
                self.weights['exploit'] = min(0.85, max(0.15, self.weights['exploit'] + delta))
                self.weights['explore'] = 1 - self.weights['exploit']
                self.save()
                return score
        except Exception as e:
            logger.warning(f"Judge failed: {e}")
        return 50

    async def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        combined = []
        for e in self.e_pool:
            combined.append({'text': e['trajectory'], 'score': e['score'],
                             'type': 'e', 'embedding': e.get('embedding')})
        for x in self.x_pool:
            combined.append({'text': x['candidate'], 'score': x['score'],
                             'type': 'x', 'embedding': x.get('embedding')})
        q_emb = await ollama_client.get_embedding(query)
        if q_emb:
            for it in combined:
                if it['embedding']:
                    it['sim'] = cosine_sim(q_emb, it['embedding'])
                else:
                    it['sim'] = 0
            combined.sort(key=lambda x: x.get('sim',0), reverse=True)
        else:
            combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:k]

    def get_stats(self) -> Dict:
        return {
            'e_pool_size': len(self.e_pool),
            'x_pool_size': len(self.x_pool),
            'judgements': self.stats.get('judgements', 0),
            'avg_score': self.stats.get('avg', 50),
            'weights': self.weights
        }

    def get_top_scores(self, k: int = 5) -> List[Tuple[str, float]]:
        combined = []
        for e in self.e_pool:
            combined.append((e['trajectory'][:100], e['score']))
        for x in self.x_pool:
            combined.append((x['candidate'][:100], x['score']))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]

# ==============================================================================
# SECTION 11: AGENT LOOP (Core orchestration + Reflection)
# PURPOSE: Processes user messages, selects appropriate agents, invokes Ollama,
#          handles skill invocation, and appends reflection.
# EXTENSION: Add multi‑turn planning, tool use, or external API calls.
# ==============================================================================
class AgentLoop:
    def __init__(self, agents: Dict, memories: Dict, skill_engine: SkillEngine, moderator, server):
        self.agents = agents
        self.memories = memories
        self.skill_engine = skill_engine
        self.moderator = moderator
        self.server = server
        self.cooldowns = defaultdict(float)
        self.context_managers = {}
        self.max_iterations = settings.max_iterations

    async def process_message(self, store_id: str, message: str, sender: str, channel_name: str = 'general') -> AsyncIterator[Dict]:
        cache_key = f"response:{store_id}:{sender}:{hashlib.sha256(message.encode()).hexdigest()[:16]}"
        cached = await cache.get(cache_key)
        if cached:
            for res in cached:
                yield res
            return

        tasks = []
        is_coding = channel_name == 'code'
        active_agents = []

        for aid, agent in self.agents.items():
            if agent.get('is_embed_operator', False):
                continue
            if channel_name not in agent.get('channels', []):
                continue
            now = time.time()
            if now - self.cooldowns[aid] < 0.5:
                continue
            self.cooldowns[aid] = now
            if self.server is not None:
                await self.server.broadcast({'type': 'agent_loading', 'agent_id': aid, 'loading': True})
            tasks.append(self._respond(aid, agent, store_id, message, sender, is_coding))
            active_agents.append(aid)

        if not tasks:
            for aid, agent in self.agents.items():
                if not agent.get('is_embed_operator', False):
                    if self.server is not None:
                        await self.server.broadcast({'type': 'agent_loading', 'agent_id': aid, 'loading': True})
                    tasks.append(self._respond(aid, agent, store_id, message, sender, is_coding))
                    active_agents.append(aid)
                    break

        responses = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                aid = active_agents[i] if i < len(active_agents) else 'unknown'
                if self.server is not None:
                    await self.server.broadcast({'type': 'agent_loading', 'agent_id': aid, 'loading': False})
                if isinstance(res, dict) and 'reply' in res:
                    res['agent_id'] = aid
                    responses.append(res)
                elif isinstance(res, Exception):
                    responses.append({'agent': f'Agent-{aid[:6]}', 'reply': f'[Agent Error] {str(res)[:100]}', 'error': True, 'agent_id': aid})

        if responses:
            await cache.set(cache_key, responses, ttl=30)
            for res in responses:
                yield res
        else:
            yield {'agent': 'System', 'reply': '[No agents available to respond.]', 'error': True, 'agent_id': 'system'}

    async def _respond(self, aid: str, agent: Dict, store_id: str, message: str, sender: str, is_coding: bool) -> Dict:
        agent_name = agent.get('name') or f"Agent-{aid[:6]}"
        if aid not in self.context_managers:
            self.context_managers[aid] = ContextManager()
        ctx = self.context_managers[aid]

        # Skill invocation
        skill_match = re.search(r'skill:\s*(\w+)', message)
        if skill_match:
            skill_name = skill_match.group(1)
            args_match = re.search(r'args:\s*\[(.*?)\]', message)
            args = [x.strip().strip('"') for x in args_match.group(1).split(',')] if args_match else []
            result = await self.skill_engine.execute(skill_name, args, aid)
            reply = f"Skill {skill_name} -> {json.dumps(result)}"
            return {'agent': agent_name, 'reply': reply}

        hello_match = re.search(r'(?:nlp/)?hello\s+(.+)', message)
        if hello_match:
            name = hello_match.group(1).strip() or 'World'
            result = await self.skill_engine.execute('hello', [name], aid)
            reply = f"Hello skill -> {json.dumps(result)}"
            return {'agent': agent_name, 'reply': reply}

        personality = get_channel_personality(store_id)
        context_str = await self._build_conversation_context(store_id, agent_name)
        system = agent.get('system_prompt', 'You are a helpful assistant.') + personality.get('system_bonus', '')
        if is_coding:
            system += "\nYou are in a coding channel. Provide code in markdown blocks."

        bio_content = read_file(settings.bio_file) if settings.bio_file.exists() else ""
        if bio_content:
            system += "\n\n---\n" + bio_content + "\n---\n"

        # Inject workspace context (J‑space) if available
        if self.server and hasattr(self.server, 'workspace_context') and self.server.workspace_context:
            context_terms = ', '.join(self.server.workspace_context)
            system += f"\n[Workspace context: {context_terms}]\n"

        prompt = f"{context_str}\n{sender}: {message}\nRespond as {agent_name}. Keep brief, high-quality."

        model = agent.get('model') or settings.default_generate_model
        degraded = False
        mem = self.memories.get(aid)
        if mem and mem.stats.get('avg', 50) < 30:
            degraded = True

        reply = await ollama_client.query(
            model, prompt, system,
            temperature=personality.get('temperature', 0.7),
            agent_id=aid, degraded=degraded
        )

        if reply.startswith('[OLLAMA_ERROR]'):
            return {'agent': agent_name, 'reply': reply}

        fixed_reply = ensure_code_block(reply, 'auto')
        ctx.add_message("user", message)
        ctx.add_message("assistant", fixed_reply)

        # Reflection (Sonnet‑style)
        reflection = await self.generate_reflection(aid, context_str + message, fixed_reply)
        if reflection:
            fixed_reply += "\n\n" + reflection

        if is_coding and '```' in fixed_reply:
            thread_id = store_id
            results = await self.moderator.process_blocks(thread_id, aid, fixed_reply)
            if results:
                feedback = self.moderator.build_feedback(aid, results, thread_id, self.moderator.ensure_repo(thread_id))
                fixed_reply += "\n\n" + feedback

        agent['status'] = 'online'
        if self.server is not None:
            await self.server.broadcast_agents()
        return {'agent': agent_name, 'reply': fixed_reply}

    async def generate_reflection(self, agent_id: str, context: str, main_reply: str) -> str:
        """Generate a reflection (things worth investigating, follow-up questions)."""
        agent = self.agents.get(agent_id)
        if not agent:
            return ""
        model = agent.get('model') or settings.default_generate_model
        prompt = f"""You just responded to the following context:
{context[:600]}

Your response was:
{main_reply[:600]}

Now reflect on your response. List 2-4 concrete items worth investigating further, and end with a clear follow‑up question. Keep it concise.
"""
        reflection = await ollama_client.query(model, prompt, temperature=0.5, agent_id=agent_id)
        if reflection.startswith('[OLLAMA_ERROR]'):
            return ""
        # Clean up
        reflection = reflection.strip()
        if not reflection:
            return ""
        # If reflection is too long, truncate
        if len(reflection) > 500:
            reflection = reflection[:500] + "..."
        return reflection

    async def _build_conversation_context(self, store_id: str, agent_name: str, max_messages: int = 8) -> str:
        try:
            rows = await db.fetchall('SELECT sender, content FROM messages WHERE store_id=? ORDER BY timestamp DESC LIMIT 100', (store_id,))
            msgs = []
            for sender, content in rows:
                if sender != agent_name and sender != 'System':
                    msgs.append(f"{sender}: {content}")
            return "\n".join(msgs[-max_messages:])
        except Exception as e:
            log_error_to_file(f"build_conversation_context: {e}")
            return ""

class ContextManager:
    def __init__(self):
        self.history = []
        self.token_count = 0
        self.max_tokens = settings.token_budget

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.token_count += len(content) // 4
        if self.token_count > self.max_tokens:
            self.compact()

    def compact(self):
        if len(self.history) > 10:
            if self.history and self.history[0]["role"] == "system":
                self.history = [self.history[0]] + self.history[-9:]
            else:
                self.history = self.history[-10:]
            self.token_count = sum(len(m["content"]) // 4 for m in self.history)

# ==============================================================================
# SECTION 12: MODERATOR (Linting, Git commits)
# PURPOSE: Extracts code blocks from agent messages, lints them (using ruff,
#          flake8, biome, etc.), and commits them to a Git repository per thread.
# EXTENSION: Add support for more languages and linters.
# ==============================================================================
class Moderator:
    def __init__(self):
        self.lint_cache = {}

    def ensure_repo(self, thread_id: str) -> Path:
        p = settings.thread_repo_dir / thread_id
        if not p.exists():
            p.mkdir(parents=True)
            if Repo:
                try:
                    repo = Repo.init(p)
                    repo.git.checkout('-b', 'main')
                    write_file(p / 'README.md', f"# Thread {thread_id}\nCreated: {time.ctime()}")
                    repo.index.add(['README.md'])
                    repo.index.commit("Initialize")
                except Exception as e:
                    logger.warning(f"Git init failed: {e}")
        return p

    async def process_blocks(self, thread_id: str, agent_id: str, text: str) -> List[Dict]:
        blocks = re.findall(r'```([a-zA-Z0-9_+#]+)\n(.*?)```', text, re.DOTALL)
        results = []
        repo_path = self.ensure_repo(thread_id)
        for i, (lang, code) in enumerate(blocks):
            lang_map = {'c++': 'cpp', 'c#': 'cs', 'js': 'javascript', 'py': 'python', 'rb': 'ruby', 'rs': 'rust'}
            lang = lang_map.get(lang.lower(), lang.lower()).strip()
            ext_map = {
                'python': 'py', 'javascript': 'js', 'json': 'json', 'html': 'html',
                'css': 'css', 'c': 'c', 'cpp': 'cpp', 'rust': 'rs', 'ruby': 'rb',
                'bash': 'sh', 'shell': 'sh'
            }
            ext = ext_map.get(lang, 'txt')
            fname = f"code_{int(time.time())}_{i}.{ext}"
            fp = repo_path / fname
            write_file(fp, code)
            errors = await self.lint(lang, code, fname)
            passed = len(errors) == 0
            if Repo:
                try:
                    repo = Repo(repo_path)
                    repo.index.add([fname])
                    msg = f"[Moderator] {fname} from {agent_id}" + ("" if passed else f" ERRORS: {'; '.join(errors)}")
                    repo.index.commit(msg)
                except Exception as e:
                    logger.warning(f"Git commit failed: {e}")
            results.append({'filename': fname, 'language': lang, 'passed': passed, 'errors': errors})
        return results

    async def lint(self, lang: str, code: str, filename: Optional[str] = None) -> List[str]:
        errors = []
        if lang == 'python':
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                errors.append(str(e))
            if shutil.which('ruff'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    tmp = f.name
                out, err, rc = await run_cmd_async(['ruff', 'check', tmp])
                if rc != 0:
                    errors.extend(out.splitlines())
                os.unlink(tmp)
            elif shutil.which('flake8'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    tmp = f.name
                out, err, rc = await run_cmd_async(['flake8', tmp])
                if rc != 0:
                    errors.extend(out.splitlines())
                os.unlink(tmp)
        elif lang == 'javascript':
            if shutil.which('biome'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                    f.write(code)
                    tmp = f.name
                out, err, rc = await run_cmd_async(['biome', 'check', tmp])
                if rc != 0:
                    errors.append(out or err)
                os.unlink(tmp)
            elif shutil.which('node'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                    f.write(code)
                    tmp = f.name
                out, err, rc = await run_cmd_async(['node', '-c', tmp])
                if rc != 0:
                    errors.append(err or out)
                os.unlink(tmp)
            else:
                if not re.search(r'^\s*(function|const|let|var|if|for|while|return)', code, re.MULTILINE):
                    errors.append("No JavaScript statements found; may be invalid.")
        elif lang == 'json':
            try:
                json.loads(code)
            except json.JSONDecodeError as e:
                errors.append(str(e))
        elif lang == 'html':
            if shutil.which('htmlhint'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    f.write(code)
                    tmp = f.name
                out, err, rc = await run_cmd_async(['htmlhint', tmp])
                if rc != 0:
                    errors.append(out or err)
                os.unlink(tmp)
            else:
                if '<' not in code or '>' not in code:
                    errors.append("No HTML tags found; may be invalid.")
        elif lang in ['c', 'cpp']:
            if shutil.which('gcc'):
                ext = '.c' if lang == 'c' else '.cpp'
                with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                    f.write(code)
                    tmp = f.name
                out, err, rc = await run_cmd_async(['gcc', '-fsyntax-only', tmp])
                if rc != 0:
                    errors.append(err or out)
                os.unlink(tmp)
            else:
                errors.append("gcc not found; C/C++ linting disabled.")
        else:
            errors.append(f"Linting for {lang} not implemented.")
        if errors:
            log_entry = f"[{datetime.datetime.now().isoformat()}] LINT ERROR in {filename or 'code-block'}: {', '.join(errors)}"
            append_to_file(settings.logs_dir / "log.md", log_entry)
            log_error_to_file(log_entry)
        return errors

    def build_feedback(self, agent_id: str, results: List[Dict], thread_id: str, repo_path: Path) -> str:
        passed_count = sum(1 for r in results if r['passed'])
        total = len(results)
        all_passed = passed_count == total
        lines = []
        lines.append(f"🛡️ **Moderator Review** (agent: {agent_id})")
        lines.append(f"Thread: `{thread_id}` | Repo: `{repo_path}`\n")
        for r in results:
            icon = '✅' if r['passed'] else '❌'
            lines.append(f"{icon} **{r['filename']}** ({r['language']})")
            if r['errors']:
                lines.append("   **Errors:**")
                for err in r['errors'][:3]:
                    lines.append(f"   - `{err[:80]}`")
            lines.append("")
        if all_passed:
            lines.append(f"🎉 **All {total} file(s) validated and committed.**")
            lines.append(f"Use `/repo {thread_id}` to browse.")
        else:
            lines.append(f"⚠️ **{total - passed_count}/{total} files failed validation.**")
            lines.append(f"@ {agent_id} please fix the errors above and re‑submit.")
        return "\n".join(lines)

# ==============================================================================
# SECTION 13: RESEARCH ENGINE (Siphon)
# PURPOSE: Performs web research: searches (DuckDuckGo, SerpApi), scrapes URLs,
#          extracts facts using LLM, and generates a summary report.
# EXTENSION: Add more search backends or scrapers.
# ==============================================================================
class ResearchEngine:
    def __init__(self, server):
        self.server = server
        self.sessions = {}
        self._model = settings.default_generate_model

    def start_siphon(self, query: str, channel: str, user: str = "Siphon") -> str:
        sid = str(uuid.uuid4())[:8]
        self.sessions[sid] = {
            'id': sid,
            'query': query,
            'channel': channel,
            'user': user,
            'phase': 'Initializing',
            'metric': 0,
            'logs': [],
            'facts': [],
            'notes': [],
            'started_at': time.time()
        }
        asyncio.create_task(self._run_siphon(sid))
        return sid

    async def _run_siphon(self, session_id: str):
        s = self.sessions[session_id]
        try:
            if not BeautifulSoup:
                s['phase'] = 'Failed (BeautifulSoup missing)'
                s['logs'].append("BeautifulSoup not installed.")
                await self._broadcast_update(session_id)
                return

            query = s['query']
            logger.info(f"Siphon starting research on: {query}")
            s['phase'] = 'Fetching content'
            s['logs'].append(f"Researching: {query}")
            await self._broadcast_update(session_id)

            urls_to_scrape = []
            facts = []

            # DuckDuckGo API
            if settings.enable_duckduckgo:
                try:
                    encoded = urllib.parse.quote(query)
                    ddg_url = f"https://api.duckduckgo.com/?q={encoded}&format=json"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(ddg_url, timeout=5,
                                               headers={'User-Agent': 'Mozilla/5.0 (compatible; OPENCHEF-SIPHON)'}) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                abstract = data.get('AbstractText', '')
                                if abstract:
                                    facts.append(f"Abstract: {abstract}")
                                    s['logs'].append("Got abstract from DuckDuckGo API")
                                topics = data.get('RelatedTopics', [])
                                for t in topics[:3]:
                                    if 'Text' in t:
                                        facts.append(t['Text'])
                                if 'Infobox' in data and 'content' in data['Infobox']:
                                    for item in data['Infobox']['content']:
                                        if 'value' in item and 'url' in item['value']:
                                            urls_to_scrape.append(item['value']['url'])
                except Exception as e:
                    s['logs'].append(f"DuckDuckGo API error: {e}")
                    logger.warning(f"Siphon DDG API error: {e}")

                # Fallback: DuckDuckGo Lite HTML
                if not facts and not urls_to_scrape:
                    try:
                        search_url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
                        async with aiohttp.ClientSession() as session:
                            async with session.get(search_url, timeout=8,
                                                   headers={'User-Agent': 'Mozilla/5.0 (compatible; OPENCHEF-SIPHON)'}) as resp:
                                if resp.status == 200:
                                    html = await resp.text()
                                    soup = BeautifulSoup(html, 'html.parser')
                                    links = [a.get('href') for a in soup.find_all('a')
                                             if a.get('href', '').startswith('http')][:5]
                                    urls_to_scrape.extend(links)
                                    s['logs'].append(f"Found {len(links)} URLs from DuckDuckGo Lite")
                    except Exception as e:
                        s['logs'].append(f"Search failed: {e}")
                        logger.warning(f"Siphon DDG Lite error: {e}")

            # SerpApi
            if settings.enable_serpapi and settings.serpapi_key:
                try:
                    encoded = urllib.parse.quote(query)
                    serp_url = f"https://serpapi.com/search.json?q={encoded}&api_key={settings.serpapi_key}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(serp_url, timeout=8) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                organic = data.get('organic_results', [])
                                for r in organic[:3]:
                                    if 'snippet' in r:
                                        facts.append(r['snippet'])
                                s['logs'].append(f"SerpApi: fetched {len(organic)} results")
                except Exception as e:
                    s['logs'].append(f"SerpApi error: {e}")
                    logger.warning(f"Siphon SerpApi error: {e}")

            # Firecrawl (placeholder)
            if settings.enable_firecrawl and settings.firecrawl_api_key:
                s['logs'].append("Firecrawl not implemented yet")

            # Scrape URLs
            if urls_to_scrape:
                s['phase'] = 'Scraping pages'
                s['metric'] = 0.2
                await self._broadcast_update(session_id)

                for idx, url in enumerate(urls_to_scrape):
                    s['logs'].append(f"Scraping {url}")
                    logger.info(f"Siphon scraping: {url}")
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, timeout=10,
                                                   headers={'User-Agent': 'Mozilla/5.0'}) as resp:
                                if resp.status == 200:
                                    html = await resp.text()
                                    soup = BeautifulSoup(html, 'html.parser')
                                    for script in soup(["script", "style"]):
                                        script.decompose()
                                    text = soup.get_text(separator=' ', strip=True)[:8000]
                                    fact_prompt = f"Extract key facts from the following text about '{query}':\n\n{text[:4000]}\n\nList each fact starting with 'FACT:'"
                                    response = await ollama_client.query(self._model, fact_prompt, temperature=0.3)
                                    if not response.startswith('[OLLAMA_ERROR]'):
                                        extracted = [line.replace('FACT:', '').strip()
                                                     for line in response.split('\n')
                                                     if line.startswith('FACT:')]
                                        facts.extend(extracted)
                                        s['logs'].append(f"Extracted {len(extracted)} facts from {url}")
                                    else:
                                        s['logs'].append(f"Ollama extraction failed for {url}")
                    except Exception as e:
                        s['logs'].append(f"Error scraping {url}: {e}")
                        logger.warning(f"Siphon scrape error {url}: {e}")

                    s['metric'] = 0.2 + 0.6 * ((idx + 1) / len(urls_to_scrape))
                    await self._broadcast_update(session_id)

            # If no facts, use LLM fallback
            if not facts:
                s['phase'] = 'Generating synthetic facts'
                s['logs'].append("No facts extracted, using LLM fallback")
                prompt = f"Generate 5 concise facts about '{query}' based on general knowledge. Each line start with FACT:"
                response = await ollama_client.query(self._model, prompt, temperature=0.5)
                if not response.startswith('[OLLAMA_ERROR]'):
                    facts = [line.replace('FACT:', '').strip()
                             for line in response.split('\n')
                             if line.startswith('FACT:')]
                    s['facts'] = facts
                    s['logs'].append(f"Generated {len(facts)} synthetic facts")

            s['facts'] = facts

            # Summarize
            s['phase'] = 'Summarizing'
            s['metric'] = 0.9
            await self._broadcast_update(session_id)

            summary = ""
            if facts:
                summary_prompt = f"Based on the following facts about '{query}', write a concise summary (3-5 sentences):\n\n" + "\n".join(facts[:20])
                summary = await ollama_client.query(self._model, summary_prompt, temperature=0.5)
                if summary.startswith('[OLLAMA_ERROR]'):
                    summary = "Summary could not be generated."

            s['phase'] = 'Complete'
            s['metric'] = 1.0
            s['notes'] = [{'question': query, 'answer': summary, 'facts': facts}]
            s['logs'].append("Research complete.")
            logger.info(f"Siphon research complete for: {query}")
            await self._broadcast_update(session_id)

            report = self._format_report(s)
            await self._post_report(session_id, report)

            report_path = settings.research_dir / f"{session_id}_{int(time.time())}.txt"
            write_file(report_path, report)
            logger.info(f"Research report saved to {report_path}")
        except Exception as e:
            log_error_to_file(f"Siphon _run_siphon: {e}")
            logger.error(f"Siphon error: {e}")
            s['phase'] = 'Error'
            s['logs'].append(f"Error: {e}")

    async def _broadcast_update(self, session_id: str):
        s = self.sessions.get(session_id)
        if not s or not self.server:
            return
        await self.server.broadcast({
            'type': 'research_update',
            'session_id': session_id,
            'data': {
                'phase': s['phase'],
                'metric': s['metric'],
                'logs': s['logs'][-5:],
                'facts_count': len(s['facts']),
                'notes_count': len(s['notes'])
            }
        })

    def _format_report(self, session: Dict) -> str:
        lines = []
        lines.append("Siphon Research Report")
        lines.append(f"Query: {session['query']}")
        lines.append(f"Status: {session['phase']}")
        if session['notes']:
            note = session['notes'][-1]
            lines.append(f"Summary: {note.get('answer', 'No summary')}")
        if session['facts']:
            lines.append(f"Key Facts ({len(session['facts'])}):")
            for fact in session['facts'][:10]:
                lines.append(f"  - {fact}")
            if len(session['facts']) > 10:
                lines.append(f"  ... and {len(session['facts'])-10} more.")
        if session['logs']:
            lines.append("Log:")
            for log in session['logs'][-3:]:
                lines.append(f"  - {log}")
        return "\n".join(lines)

    async def _post_report(self, session_id: str, report: str):
        s = self.sessions[session_id]
        if not s or not self.server:
            return
        msg = {
            'id': str(uuid.uuid4()),
            'store_id': 'siphon',
            'sender': 'Siphon',
            'sender_type': 'agent',
            'content': report,
            'parent_id': '',
            'thread_id': '',
            'timestamp': int(time.time() * 1000),
            'reactions': {},
            'pinned': False
        }
        await db.execute(
            'INSERT OR REPLACE INTO messages (id, store_id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (msg['id'], msg['store_id'], msg['sender'], msg['sender_type'], msg['content'],
             msg['parent_id'], msg['thread_id'], msg['timestamp'], json.dumps(msg['reactions']), int(msg['pinned']))
        )
        await self.server.broadcast({'type': 'new_message', 'channel': 'siphon', 'message': msg})

# ==============================================================================
# SECTION 14: DATASET ENGINE
# PURPOSE: Parses dataset specification files (Markdown with YAML frontmatter)
#          and runs extraction jobs: scrapes URLs, extracts fields using LLM,
#          and stores results in vector DB or JSON.
# EXTENSION: Add more extraction strategies or output formats.
# ==============================================================================
class DatasetEngine:
    def __init__(self, server):
        self.server = server
        self.active_jobs = {}
        self.job_progress = {}
        self._model = settings.default_generate_model
        self._chroma_available = False
        if chromadb:
            try:
                self.chroma_client = chromadb.Client()
                self._chroma_available = True
            except Exception as e:
                logger.warning(f"Could not initialize chromadb: {e}")
                self.chroma_client = None
        else:
            self.chroma_client = None

    def parse_dataset_file(self, path: Path) -> Dict[str, Any]:
        content = read_file(path)
        if not content:
            raise ValueError(f"Empty or unreadable file: {path}")

        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        metadata = {}
        if frontmatter_match and yaml:
            try:
                metadata = yaml.safe_load(frontmatter_match.group(1))
            except Exception as e:
                logger.warning(f"Failed to parse YAML frontmatter: {e}")
        dataset_id = metadata.get('dataset_id', path.stem)
        embed_model = metadata.get('target_embedding_model', settings.default_embed_model)
        dimension = metadata.get('dimension', 768)
        chunk_size = metadata.get('chunk_size', 512)
        chunk_overlap = metadata.get('chunk_overlap', 64)
        storage_format = metadata.get('storage_format', 'json')

        table_pattern = r'\| Field Name \| Data Type \| Description \| Extraction Strategy / Selector \|\s*\n\|[:\- ]+\|[:\- ]+\|[:\- ]+\|[:\- ]+\|\s*\n((?:\|.*\|\s*\n)+)'
        table_match = re.search(table_pattern, content)
        schema_fields = []
        if table_match:
            rows = table_match.group(1).strip().split('\n')
            for row in rows:
                cols = [c.strip() for c in row.split('|')[1:-1]]
                if len(cols) >= 4:
                    field = {
                        'name': cols[0].strip('`'),
                        'type': cols[1],
                        'description': cols[2],
                        'selector': cols[3]
                    }
                    schema_fields.append(field)

        targets = []
        url_blocks = re.findall(r'- \[ \] URL: `([^`]+)`\s*(?:- DOM_Hint: `([^`]*)`\s*)?(?:- Priority: (\w+)\s*)?', content, re.DOTALL)
        for url, dom_hint, priority in url_blocks:
            targets.append({
                'url': url.strip(),
                'dom_hint': dom_hint.strip() if dom_hint else None,
                'priority': priority.strip() if priority else 'Medium'
            })

        return {
            'dataset_id': dataset_id,
            'embed_model': embed_model,
            'dimension': dimension,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'storage_format': storage_format,
            'schema_fields': schema_fields,
            'targets': targets,
            'file_path': path
        }

    async def run_dataset_job(self, dataset_name: str, user: str = "System") -> str:
        job_id = str(uuid.uuid4())[:8]
        possible_paths = [
            settings.datasets_dir / f"{dataset_name}.md",
            settings.base_dir / f"{dataset_name}.md",
            settings.base_dir / "datasets.md"
        ]
        file_path = None
        for p in possible_paths:
            if p.exists():
                file_path = p
                break
        if not file_path:
            raise FileNotFoundError(f"Dataset file not found for: {dataset_name}")

        dataset_spec = self.parse_dataset_file(file_path)
        self.job_progress[job_id] = {
            'status': 'initializing',
            'dataset': dataset_spec['dataset_id'],
            'total_targets': len(dataset_spec['targets']),
            'processed': 0,
            'errors': 0,
            'entries': [],
            'started_at': time.time(),
            'logs': []
        }

        task = asyncio.create_task(self._run_job(job_id, dataset_spec))
        self.active_jobs[job_id] = task
        await self._broadcast_progress(job_id)
        return job_id

    async def _run_job(self, job_id: str, dataset_spec: Dict):
        progress = self.job_progress[job_id]
        progress['status'] = 'running'
        await self._broadcast_progress(job_id)

        collection_name = f"dataset_{dataset_spec['dataset_id']}"
        if self._chroma_available and self.chroma_client:
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = None
            file_store = settings.datasets_dir / f"{dataset_spec['dataset_id']}.json"
            if not file_store.exists():
                file_store.write_text('[]')

        schema_fields = dataset_spec['schema_fields']
        schema_description = "\n".join([f"- {f['name']} ({f['type']}): {f['description']}" for f in schema_fields])
        targets = dataset_spec['targets']

        for idx, target in enumerate(targets):
            progress['logs'].append(f"Scraping {target['url']}")
            await self._broadcast_progress(job_id)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(target['url'], timeout=10,
                                           headers={'User-Agent': 'Mozilla/5.0 (compatible; OPENCHEF-DATASET)'}) as resp:
                        if resp.status != 200:
                            raise Exception(f"HTTP {resp.status}")
                        html = await resp.text()
            except Exception as e:
                progress['errors'] += 1
                progress['logs'].append(f"Error fetching {target['url']}: {e}")
                await self._broadcast_progress(job_id)
                continue

            if BeautifulSoup:
                soup = BeautifulSoup(html, 'html.parser')
                if target['dom_hint']:
                    elements = soup.select(target['dom_hint'])
                    if elements:
                        for elem in elements:
                            for script in elem(["script", "style"]):
                                script.decompose()
                        text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    else:
                        text = soup.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text).strip()[:8000]
            else:
                text = "BeautifulSoup not available"

            extraction_prompt = f"""You are a data extraction agent. Extract the following fields from the provided webpage content according to the schema.

Schema:
{schema_description}

Webpage content:
{text}

Return a JSON object with exactly the fields listed above. Use null if a field cannot be extracted.
Do not include any extra text, reasoning, or markdown.
"""
            extracted = None
            for attempt in range(3):
                try:
                    response = await ollama_client.query(self._model, extraction_prompt, temperature=0.0)
                    parsed = extract_json(response)
                    if parsed:
                        required_fields = [f['name'] for f in schema_fields]
                        missing = [f for f in required_fields if f not in parsed]
                        if missing:
                            progress['logs'].append(f"Attempt {attempt+1}: Missing fields {missing}, retrying...")
                            extraction_prompt += f"\n\nThe previous attempt failed to extract fields: {missing}. Please ensure all fields are present. Return valid JSON."
                            continue
                        extracted = parsed
                        break
                    else:
                        progress['logs'].append(f"Attempt {attempt+1}: Failed to parse JSON, retrying...")
                except Exception as e:
                    progress['logs'].append(f"Ollama extraction error: {e}")

            if extracted is None:
                progress['errors'] += 1
                progress['logs'].append(f"Failed to extract from {target['url']} after retries")
                await self._broadcast_progress(job_id)
                continue

            extracted['source_url'] = target['url']
            extracted['_extracted_at'] = datetime.datetime.utcnow().isoformat()
            extracted['_id'] = str(uuid.uuid4())

            if collection:
                text_rep = json.dumps(extracted, sort_keys=True)
                embedding = await ollama_client.get_embedding(text_rep)
                if embedding:
                    collection.add(
                        ids=[extracted['_id']],
                        embeddings=[embedding],
                        metadatas=[extracted],
                        documents=[text_rep]
                    )
                else:
                    progress['logs'].append("Embedding failed, storing without vector")
            else:
                file_store = settings.datasets_dir / f"{dataset_spec['dataset_id']}.json"
                try:
                    existing = json.loads(file_store.read_text())
                except:
                    existing = []
                existing.append(extracted)
                file_store.write_text(json.dumps(existing, indent=2))

            output_file = settings.datasets_dir / f"{dataset_spec['dataset_id']}_{int(time.time())}.jsonl"
            with output_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(extracted) + "\n")

            progress['processed'] += 1
            progress['entries'].append(extracted)
            progress['logs'].append(f"Extracted {len(extracted)} fields from {target['url']}")
            await self._broadcast_progress(job_id)

        progress['status'] = 'complete'
        progress['logs'].append("Dataset job finished.")
        await self._broadcast_progress(job_id)

    async def _broadcast_progress(self, job_id: str):
        progress = self.job_progress.get(job_id)
        if not progress or not self.server:
            return
        await self.server.broadcast({
            'type': 'dataset_progress',
            'job_id': job_id,
            'data': {
                'status': progress['status'],
                'total': progress['total_targets'],
                'processed': progress['processed'],
                'errors': progress['errors'],
                'logs': progress['logs'][-10:],
                'entries_count': len(progress['entries'])
            }
        })

# ==============================================================================
# SECTION 15: CRON ENGINE
# PURPOSE: Schedules and executes periodic jobs using cron expressions.
#          Relies on croniter if installed; otherwise runs every hour.
# EXTENSION: Add more job types beyond webhooks.
# ==============================================================================
class CronEngine:
    def __init__(self, server):
        self.server = server
        self.running = False
        self._task = None

    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        while self.running:
            try:
                await self._check_jobs()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron loop error: {e}")
                await asyncio.sleep(60)

    async def _check_jobs(self):
        rows = await db.fetchall('SELECT * FROM cron_jobs WHERE enabled=1')
        now = int(time.time())
        for row in rows:
            name, schedule, webhook_name, enabled, last_run, next_run, created_at = row
            if next_run and next_run <= now:
                await self._execute_job(name, webhook_name)
                if croniter:
                    base = datetime.datetime.fromtimestamp(now)
                    try:
                        iter = croniter(schedule, base)
                        next_run = iter.get_next(int)
                    except Exception as e:
                        logger.error(f"Cron schedule parse error for {name}: {e}")
                        next_run = now + 3600
                else:
                    next_run = now + 3600
                await db.execute('UPDATE cron_jobs SET last_run=?, next_run=? WHERE name=?', (now, next_run, name))

    async def _execute_job(self, job_name: str, webhook_name: str):
        rows = await db.fetchall('SELECT * FROM webhooks WHERE name=?', (webhook_name,))
        if rows:
            row = rows[0]
            name, url, method, headers, created_at = row
            try:
                async with aiohttp.ClientSession() as session:
                    headers_dict = json.loads(headers) if headers else {}
                    async with session.request(method, url, headers=headers_dict) as resp:
                        result = await resp.text()
                        logger.info(f"Cron job {job_name} executed: {resp.status}")
                        await self.server.broadcast({
                            'type': 'cron_result',
                            'job': job_name,
                            'status': resp.status,
                            'result': result[:500]
                        })
            except Exception as e:
                logger.error(f"Cron job {job_name} failed: {e}")

# ==============================================================================
# SECTION 15b: STACK ENGINE (Template Management)
# PURPOSE: Manages reusable templates/skills (STACK system). Can build repos
#          from templates, add new templates by description, and bulk import.
# EXTENSION: Add semantic search over templates.
# ==============================================================================
class StackEngine:
    """Reusable template/skill injection system (LACK STACK)."""
    def __init__(self):
        self.templates = {}  # name -> content

    async def load_templates(self):
        """Load templates from stack_templates directory and DB."""
        self.templates.clear()
        # Load from files
        for f in settings.stack_dir.glob("*"):
            if f.is_file():
                self.templates[f.stem] = read_file(f)
        # Load from DB
        rows = await db.fetchall('SELECT name, content FROM stack_templates')
        for name, content in rows:
            self.templates[name] = content
        logger.info(f"Loaded {len(self.templates)} stack templates")

    async def save_template(self, name: str, content: str, description: str = "", tags: str = ""):
        self.templates[name] = content
        # write to file
        (settings.stack_dir / f"{name}.template").write_text(content, encoding='utf-8')
        # save to DB
        await db.execute(
            'INSERT OR REPLACE INTO stack_templates (name, content, description, tags, created_at) VALUES (?,?,?,?,?)',
            (name, content, description, tags, int(time.time()))
        )

    async def build(self, name: str, target_dir: Path) -> bool:
        """Inject template into target directory (e.g., create a new repo)."""
        if name not in self.templates:
            return False
        content = self.templates[name]
        target_dir.mkdir(parents=True, exist_ok=True)
        # If content is a directory structure encoded as markdown, parse it.
        # For simplicity, we just write a single file.
        (target_dir / "README.md").write_text(f"# {name}\n\n{content}", encoding='utf-8')
        return True

    async def add(self, description: str, agent_id: Optional[str] = None) -> str:
        """Semantic search over templates and inject matching into workspace."""
        # For now, just create a new template from description
        prompt = f"Create a reusable template/skill based on: {description}. Output the template content only, no explanations."
        response = await ollama_client.query(settings.default_generate_model, prompt, temperature=0.3, agent_id=agent_id)
        if response.startswith('[OLLAMA_ERROR]'):
            return f"Error: {response}"
        # Generate a name from description
        name = re.sub(r'[^a-zA-Z0-9_]', '_', description[:30])
        # Save template
        await self.save_template(name, response, description=description)
        return f"Template '{name}' created and added."

    async def import_(self, json_data: str) -> int:
        """Bulk import templates from JSON."""
        try:
            data = json.loads(json_data)
        except:
            return -1
        count = 0
        for item in data:
            if 'name' in item and 'content' in item:
                await self.save_template(item['name'], item['content'], item.get('description', ''), item.get('tags', ''))
                count += 1
        return count

# ==============================================================================
# SECTION 15c: RALPH ENGINE (Evolutionary Brainstorming)
# PURPOSE: Evolves a project specification over multiple generations using
#          LLM‑driven mutation, evaluation, and convergence detection.
# EXTENSION: Adjust evaluation metrics or add diversity enforcement.
# ==============================================================================
class RalphEngine:
    def __init__(self, server, agent_loop):
        self.server = server
        self.agent_loop = agent_loop
        self.sessions = {}
        self._model = settings.default_generate_model

    def start_ralph(self, topic: str, store_id: str = 'ralph') -> str:
        sid = str(uuid.uuid4())[:8]
        self.sessions[sid] = {
            'id': sid,
            'topic': topic,
            'store_id': store_id,
            'current_gen': 0,
            'max_generations': settings.max_generations,
            'convergence': 0.0,
            'stagnation': 0,
            'status': 'running',
            'best_spec': '',
            'population': [],  # list of specs
            'started_at': time.time(),
            'logs': []
        }
        asyncio.create_task(self._run_ralph(sid))
        return sid

    async def _run_ralph(self, session_id: str):
        s = self.sessions[session_id]
        try:
            topic = s['topic']
            logger.info(f"Ralph starting evolution on: {topic}")
            s['logs'].append(f"Starting Ralph on: {topic}")
            await self._broadcast_update(session_id)

            # Initial generation: generate population_size specs
            pop_size = settings.population_size
            population = []
            for i in range(pop_size):
                spec = await self._generate_spec(topic, seed=i)
                if spec:
                    population.append(spec)
            if not population:
                s['status'] = 'failed'
                s['logs'].append("No initial specs generated.")
                await self._broadcast_update(session_id)
                return

            s['population'] = population
            s['best_spec'] = population[0]  # initial best
            s['current_gen'] = 0

            while s['current_gen'] < s['max_generations']:
                s['current_gen'] += 1
                s['logs'].append(f"Generation {s['current_gen']}")

                # Evaluate population: for each spec, compute quality (use a judge or LLM score)
                scored = []
                for spec in population:
                    score = await self._evaluate_spec(spec, topic)
                    scored.append((spec, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                best_spec, best_score = scored[0]
                s['best_spec'] = best_spec

                # Update convergence: compare best to previous best
                prev_best = s.get('_prev_best', '')
                if prev_best:
                    emb1 = await ollama_client.get_embedding(prev_best)
                    emb2 = await ollama_client.get_embedding(best_spec)
                    if emb1 and emb2:
                        sim = cosine_sim(emb1, emb2)
                        s['convergence'] = sim
                        if sim >= settings.convergence_threshold:
                            s['logs'].append(f"Convergence reached: {sim:.3f}")
                            break
                        # stagnation check
                        if sim > 0.9:
                            s['stagnation'] += 1
                        else:
                            s['stagnation'] = 0
                        if s['stagnation'] >= settings.stagnation_limit:
                            s['logs'].append("Stagnation detected, forcing mutation.")
                            # Force mutation: generate a new spec with higher temperature
                            new_spec = await self._generate_spec(topic, seed=time.time(), temperature=1.2)
                            if new_spec:
                                population.append(new_spec)
                                s['stagnation'] = 0
                s['_prev_best'] = best_spec

                # Evolve: generate next generation by mutating best
                new_pop = []
                for _ in range(pop_size):
                    mutated = await self._mutate_spec(best_spec, topic)
                    if mutated:
                        new_pop.append(mutated)
                if new_pop:
                    population = new_pop
                else:
                    # fallback: keep best
                    population = [best_spec] + [best_spec] * (pop_size-1)

                s['population'] = population
                # Post report for this generation
                report = f"Ralph Gen {s['current_gen']}: {best_spec[:200]}...\nScore: {best_score:.2f}"
                await self._post_report(session_id, report)

                await self._broadcast_update(session_id)
                await asyncio.sleep(2)  # throttle

            s['status'] = 'complete'
            s['logs'].append("Ralph finished.")
            final_report = f"Ralph Final (Gen {s['current_gen']}):\n{s['best_spec']}"
            await self._post_report(session_id, final_report)
            await self._broadcast_update(session_id)
            logger.info(f"Ralph session {session_id} complete.")
        except Exception as e:
            log_error_to_file(f"Ralph _run_ralph: {e}")
            logger.error(f"Ralph error: {e}")
            s['status'] = 'error'
            s['logs'].append(f"Error: {e}")
            await self._broadcast_update(session_id)

    async def _generate_spec(self, topic: str, seed: Any = None, temperature: float = 0.7) -> str:
        prompt = f"Generate a detailed specification for: {topic}. Be concrete and comprehensive."
        if seed is not None:
            prompt += f"\nSeed: {seed}"
        response = await ollama_client.query(self._model, prompt, temperature=temperature)
        if response.startswith('[OLLAMA_ERROR]'):
            return ""
        return response.strip()

    async def _mutate_spec(self, spec: str, topic: str) -> str:
        prompt = f"Improve the following specification for '{topic}'. Add clarity, detail, and new ideas. Return only the improved spec.\n\n{spec}"
        response = await ollama_client.query(self._model, prompt, temperature=0.8)
        if response.startswith('[OLLAMA_ERROR]'):
            return ""
        return response.strip()

    async def _evaluate_spec(self, spec: str, topic: str) -> float:
        prompt = f"Rate this specification for '{topic}' on a scale 0-100. Return only a number.\n\n{spec}"
        response = await ollama_client.query(self._model, prompt, temperature=0.2)
        try:
            score = float(response.strip())
            return min(100, max(0, score))
        except:
            return 50.0

    async def _post_report(self, session_id: str, report: str):
        s = self.sessions[session_id]
        if not s or not self.server:
            return
        msg = {
            'id': str(uuid.uuid4()),
            'store_id': s['store_id'],
            'sender': 'Ralph',
            'sender_type': 'agent',
            'content': report,
            'parent_id': '',
            'thread_id': '',
            'timestamp': int(time.time() * 1000),
            'reactions': {},
            'pinned': False
        }
        await db.execute(
            'INSERT OR REPLACE INTO messages (id, store_id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (msg['id'], msg['store_id'], msg['sender'], msg['sender_type'], msg['content'],
             msg['parent_id'], msg['thread_id'], msg['timestamp'], json.dumps(msg['reactions']), int(msg['pinned']))
        )
        await self.server.broadcast({'type': 'new_message', 'channel': s['store_id'], 'message': msg})

    async def _broadcast_update(self, session_id: str):
        s = self.sessions.get(session_id)
        if not s or not self.server:
            return
        await self.server.broadcast({
            'type': 'ralph_update',
            'session_id': session_id,
            'data': {
                'current_gen': s['current_gen'],
                'convergence': s['convergence'],
                'stagnation': s['stagnation'],
                'status': s['status'],
                'best_spec': s['best_spec'][:300],
                'logs': s['logs'][-5:]
            }
        })

    def get_sessions(self) -> List[Dict]:
        return [{
            'id': sid,
            'goal': s['topic'],
            'current_gen': s['current_gen'],
            'convergence': s['convergence'],
            'stagnation': s['stagnation'],
            'status': s['status']
        } for sid, s in self.sessions.items()]

# ==============================================================================
# SECTION 15d: ABSTRACT ENGINE (Planning)
# PURPOSE: Generates structured step‑by‑step plans for user queries using an LLM.
# EXTENSION: Add plan execution or tracking.
# ==============================================================================
class AbstractEngine:
    def __init__(self, server):
        self.server = server
        self._model = settings.default_generate_model

    async def generate_plan(self, query: str, store_id: str) -> Dict:
        """Generate a structured plan and post it as a message."""
        prompt = f"""You are a planning agent. Create a detailed step‑by‑step plan for the following request.

Request: {query}

Your plan must include:
- Goal
- Prerequisites
- Steps (numbered)
- Risks and mitigation
- Expected outcome

Return the plan as a structured text. Use markdown headings and bullet points.
"""
        response = await ollama_client.query(self._model, prompt, temperature=0.4)
        if response.startswith('[OLLAMA_ERROR]'):
            return {"error": response}

        # Post as message
        msg = {
            'id': str(uuid.uuid4()),
            'store_id': store_id,
            'sender': 'Abstract',
            'sender_type': 'agent',
            'content': f"## Abstract Plan for: {query}\n\n{response}",
            'parent_id': '',
            'thread_id': '',
            'timestamp': int(time.time() * 1000),
            'reactions': {},
            'pinned': False
        }
        await db.execute(
            'INSERT OR REPLACE INTO messages (id, store_id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (msg['id'], msg['store_id'], msg['sender'], msg['sender_type'], msg['content'],
             msg['parent_id'], msg['thread_id'], msg['timestamp'], json.dumps(msg['reactions']), int(msg['pinned']))
        )
        if self.server:
            await self.server.broadcast({'type': 'new_message', 'channel': store_id, 'message': msg})
        return msg

# ==============================================================================
# SECTION 16: DASHBOARD SERVER (Web UI + WebSocket + Background Tasks)
# PURPOSE: Main HTTP and WebSocket server. Manages agents, broadcasts metrics,
#          handles commands, and coordinates all engines.
# EXTENSION: Add new API endpoints or background tasks.
# ==============================================================================
class DashboardServer:
    def __init__(self):
        self.skill_engine = SkillEngine()
        self.moderator = Moderator()
        self.research_engine = ResearchEngine(self)
        self.dataset_engine = DatasetEngine(self)
        self.cron_engine = CronEngine(self)
        self.stack_engine = StackEngine()
        self.abstract_engine = AbstractEngine(self)
        self.channels = ['general', 'code', 'siphon', 'ralph']
        self.agents = {}
        self.memories = {}
        self.websockets = set()
        self.bio_manager = BioManager()
        self.agent_loop = None
        self.ralph_engine = None
        self.agent_metrics = {}
        self.metrics_history = defaultdict(lambda: {'timestamps': [], 'response_times': [], 'memory_scores': [], 'activity': []})
        self._metrics_broadcast_task = None
        self._backup_task = None
        self._proactive_task = None
        self._shutdown_event = asyncio.Event()
        self.global_metrics = {'cpu': 0.0, 'memory': 0.0, 'token_rate': 0.0}
        self._psutil_lock = asyncio.Lock()
        # J‑space workspace context
        self.workspace_context = []  # list of concepts
        # Model cache for dropdown
        self._cached_models = []
        self._models_cache_time = 0

    async def _load_agents(self):
        rows = await db.fetchall('SELECT * FROM agents')
        if rows:
            for row in rows:
                try:
                    channels = json.loads(row[4]) if row[4] else ['general']
                    if not isinstance(channels, list) or not channels:
                        channels = ['general']
                except (json.JSONDecodeError, TypeError):
                    channels = ['general']
                agent = {
                    'id': row[0],
                    'name': row[1] or f"Agent-{str(row[0])[:6]}",
                    'model': row[2] or settings.default_generate_model,
                    'system_prompt': row[3] or 'You are a helpful assistant.',
                    'channels': channels,
                    'strict_channel': row[5],
                    'status': row[6] or 'online',
                    'is_embed_operator': bool(row[7]) if row[7] is not None else False,
                    'is_code_moderator': bool(row[8]) if row[8] is not None else False,
                    'last_response_time': {}
                }
                self.agents[row[0]] = agent
        else:
            default_agents = [
                {'id': 'agent1', 'name': 'Agent 1', 'model': settings.default_generate_model,
                 'system_prompt': 'You are a helpful assistant.', 'channels': self.channels, 'strict_channel': None},
                {'id': 'agent2', 'name': 'Agent 2', 'model': settings.default_generate_model,
                 'system_prompt': 'You are a creative problem solver.', 'channels': self.channels, 'strict_channel': None},
                {'id': 'moderator', 'name': 'Moderator', 'model': settings.default_embed_model,
                 'system_prompt': 'Embedding only.', 'channels': ['code'], 'strict_channel': None,
                 'is_embed_operator': True, 'is_code_moderator': True}
            ]
            for a in default_agents:
                self.agents[a['id']] = a
                await db.execute(
                    'INSERT OR REPLACE INTO agents (id, name, model, system_prompt, channels, strict_channel, status, is_embed_operator, is_code_moderator) VALUES (?,?,?,?,?,?,?,?,?)',
                    (a['id'], a['name'], a['model'], a['system_prompt'],
                     json.dumps(a.get('channels', [])), a.get('strict_channel'),
                     a.get('status', 'online'), int(a.get('is_embed_operator', 0)), int(a.get('is_code_moderator', 0)))
                )

        self.memories = {aid: DecentMem(aid) for aid in self.agents}
        self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, self)
        self.ralph_engine = RalphEngine(self, self.agent_loop)

        for aid in self.agents:
            if aid != 'moderator':
                self.metrics_history[aid] = {'timestamps': [], 'response_times': [], 'memory_scores': [], 'activity': []}
                self.agent_metrics[aid] = self._generate_initial_metrics()

    def _generate_initial_metrics(self):
        now = time.time() * 1000
        return {
            'cpu': [10 + i % 20 for i in range(60)],
            'mem': [20 + (i % 15) for i in range(60)],
            'activity': [30 + (i % 30) for i in range(60)],
            'timestamps': [now - (59 - i) * 3000 for i in range(60)]
        }

    async def _refresh_agent_status_loop(self):
        while True:
            await asyncio.sleep(30)
            for aid, agent in self.agents.items():
                if agent.get('status') != 'offline':
                    agent['status'] = 'online'
            await self.broadcast_agents()

    async def _global_metrics_loop(self):
        while True:
            await asyncio.sleep(2)
            cpu = 0.0
            mem = 0.0
            if psutil:
                try:
                    cpu = await asyncio.to_thread(psutil.cpu_percent)
                    mem = await asyncio.to_thread(lambda: psutil.virtual_memory().percent)
                except Exception as e:
                    logger.warning(f"psutil error: {e}")
            async with ollama_client._token_lock:
                now = time.time()
                if now - ollama_client.last_token_time > 0:
                    rate = ollama_client.token_count / (now - ollama_client.last_token_time)
                else:
                    rate = 0
                ollama_client.token_count = 0
                ollama_client.last_token_time = now
            self.global_metrics = {'cpu': cpu, 'memory': mem, 'token_rate': rate}
            await self.broadcast({'type': 'global_metrics', 'data': self.global_metrics})

    async def _lint_app_loop(self):
        while True:
            await asyncio.sleep(3600)
            if shutil.which('ruff'):
                out, err, rc = await run_cmd_async(['ruff', 'check', str(settings.base_dir / 'openchef.py')])
                if rc != 0:
                    log_error_to_file(f"Lint issues found:\n{out}\n{err}")
            elif shutil.which('flake8'):
                out, err, rc = await run_cmd_async(['flake8', str(settings.base_dir / 'openchef.py')])
                if rc != 0:
                    log_error_to_file(f"Lint issues found:\n{out}\n{err}")

    async def _proactive_question_loop(self):
        """Periodically trigger agents to ask questions in #general."""
        while True:
            await asyncio.sleep(settings.proactive_question_interval)
            if not self.agents:
                continue
            # Pick up to 2 random agents (excluding moderator)
            agent_ids = [aid for aid in self.agents if aid != 'moderator']
            if not agent_ids:
                continue
            chosen = random.sample(agent_ids, min(2, len(agent_ids)))
            for aid in chosen:
                agent = self.agents[aid]
                # Generate a question about recent conversation
                # Get last few messages from general
                rows = await db.fetchall('SELECT content FROM messages WHERE store_id="general" ORDER BY timestamp DESC LIMIT 5')
                recent = "\n".join([row[0] for row in rows]) if rows else "No recent conversation."
                prompt = f"Based on recent conversation: {recent}\nGenerate a short, relevant question to ask the group. Keep it concise."
                question = await ollama_client.query(agent['model'], prompt, temperature=0.7, agent_id=aid)
                if question and not question.startswith('[OLLAMA_ERROR]'):
                    # Post as message from agent
                    msg = {
                        'id': str(uuid.uuid4()),
                        'store_id': 'general',
                        'sender': agent['name'],
                        'sender_type': 'agent',
                        'content': f"🤔 {question[:200]}",
                        'parent_id': '',
                        'thread_id': '',
                        'timestamp': int(time.time() * 1000),
                        'reactions': {},
                        'pinned': 0
                    }
                    await db.execute(
                        'INSERT OR REPLACE INTO messages (id, store_id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (msg['id'], msg['store_id'], msg['sender'], msg['sender_type'], msg['content'],
                         msg['parent_id'], msg['thread_id'], msg['timestamp'], json.dumps(msg['reactions']), msg['pinned'])
                    )
                    await self.broadcast({'type': 'new_message', 'channel': 'general', 'message': msg})

    async def _workspace_context_loop(self):
        """Maintain workspace context (J‑space) from recent messages."""
        while True:
            await asyncio.sleep(60)  # update every minute
            try:
                rows = await db.fetchall('SELECT content FROM messages ORDER BY timestamp DESC LIMIT 20')
                texts = [row[0] for row in rows if row[0]]
                if texts:
                    combined = "\n".join(texts)[:2000]
                    # Extract key concepts using a simple LLM call
                    prompt = f"Extract up to {settings.workspace_context_size} key concepts from the following text. Return as a comma-separated list:\n\n{combined}"
                    response = await ollama_client.query(settings.default_generate_model, prompt, temperature=0.2)
                    if not response.startswith('[OLLAMA_ERROR]'):
                        concepts = [c.strip() for c in response.split(',') if c.strip()]
                        self.workspace_context = concepts[:settings.workspace_context_size]
                        logger.debug(f"Workspace context updated: {self.workspace_context}")
            except Exception as e:
                logger.warning(f"Workspace context update error: {e}")

    async def start(self):
        await db.init()
        await self._load_agents()
        await self.skill_engine.reload()
        await self.stack_engine.load_templates()
        logger.info(f"Loaded skills: {list(self.skill_engine.registry.keys())}")
        logger.info(f"Loaded stack templates: {list(self.stack_engine.templates.keys())}")
        await self.cron_engine.start()

        asyncio.create_task(self._refresh_agent_status_loop())
        asyncio.create_task(self._global_metrics_loop())
        asyncio.create_task(self._lint_app_loop())
        asyncio.create_task(self._proactive_question_loop())
        asyncio.create_task(self._workspace_context_loop())

        app = web.Application()
        app.router.add_get('/', self.handle_index)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_get('/api/metrics', self.handle_metrics)
        app.router.add_get('/api/tags', self.handle_models)
        app.router.add_get('/api/errorlog', self.handle_errorlog)
        app.router.add_get('/health', self.handle_health)
        app.router.add_post('/api/heartbeat', self.handle_heartbeat)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, settings.host, settings.port)
        try:
            await site.start()
        except OSError as e:
            if e.errno == 98:
                logger.error(f"Port {settings.port} is already in use.")
                sys.exit(1)
            else:
                raise
        url = f"http://{settings.host}:{settings.port}"
        logger.info(f"Dashboard: {url}")
        webbrowser.open(url)

        self._metrics_broadcast_task = asyncio.create_task(self._broadcast_metrics_loop())
        self._backup_task = asyncio.create_task(self._backup_loop())

        await self._shutdown_event.wait()

        logger.info("Shutting down server...")
        await self.cron_engine.stop()
        if self._metrics_broadcast_task:
            self._metrics_broadcast_task.cancel()
            try:
                await self._metrics_broadcast_task
            except asyncio.CancelledError:
                pass
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        if self._proactive_task:
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass
        await runner.cleanup()
        await db.close()
        logger.info("Server shutdown complete.")

    async def _broadcast_metrics_loop(self):
        while True:
            await asyncio.sleep(settings.metrics_broadcast_interval)
            await self.broadcast_metrics()

    async def _backup_loop(self):
        while True:
            try:
                await asyncio.sleep(settings.backup_interval_hours * 3600)
                await self.backup_database()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup loop error: {e}")

    async def backup_database(self):
        backup_dir = settings.backups_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"factory_{timestamp}.db"
        try:
            async with aiosqlite.connect(str(settings.db_path)) as src:
                await src.backup(backup_path)
            backups = sorted(backup_dir.glob("factory_*.db"))
            for old_backup in backups[:-10]:
                old_backup.unlink()
            logger.info(f"Database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Database backup failed: {e}")

    async def broadcast_metrics(self):
        try:
            metrics = {}
            for aid, agent in self.agents.items():
                if aid == 'moderator':
                    continue

                if aid not in self.agent_metrics:
                    self.agent_metrics[aid] = self._generate_initial_metrics()

                m = self.agent_metrics[aid]

                try:
                    cpu = max(1, min(95, m['cpu'][-1] + random.randint(-8, 8)) if m['cpu'] else 30)
                    mem = max(1, min(95, m['mem'][-1] + random.randint(-5, 5)) if m['mem'] else 40)
                    activity = max(1, min(100, m['activity'][-1] + random.randint(-15, 15)) if m['activity'] else 50)
                except (IndexError, KeyError):
                    cpu, mem, activity = 30, 40, 50

                m['cpu'] = (m['cpu'][1:] + [cpu]) if m.get('cpu') else [cpu] * 60
                m['mem'] = (m['mem'][1:] + [mem]) if m.get('mem') else [mem] * 60
                m['activity'] = (m['activity'][1:] + [activity]) if m.get('activity') else [activity] * 60
                m['timestamps'] = (m['timestamps'][1:] + [int(time.time() * 1000)]) if m.get('timestamps') else [int(time.time() * 1000)] * 60

                self.agent_metrics[aid] = m

                mem_obj = self.memories.get(aid)
                memory_score = mem_obj.stats.get('avg', 50) if mem_obj else 50

                metrics[aid] = {
                    'cpu': m.get('cpu', []),
                    'mem': m.get('mem', []),
                    'activity': m.get('activity', []),
                    'timestamps': m.get('timestamps', []),
                    'response_time': random.uniform(0.2, 1.5),
                    'memory_score': memory_score,
                    'agent_name': agent.get('name', aid)
                }

            if metrics:
                await self.broadcast({'type': 'metrics_update', 'agents': metrics})
        except Exception as e:
            logger.error(f"Metrics broadcast error: {e}")
            log_error_to_file(f"Metrics broadcast error: {e}")

    async def handle_index(self, request):
        html = INDEX_HTML.replace('{{WS_PORT}}', str(settings.port))
        return web.Response(text=html, content_type='text/html')

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse(
            heartbeat=settings.websocket_ping_interval,
            max_msg_size=settings.max_message_size
        )
        await ws.prepare(request)
        self.websockets.add(ws)

        client_state = {
            'channel': 'general',
            'username': 'human',
            'last_ping': time.time(),
            'closed': False
        }

        send_queue = asyncio.Queue(maxsize=100)
        ws._send_queue = send_queue

        async def reader():
            try:
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            await self._handle_ws_message(ws, data, client_state)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON: {msg.data[:100]}")
                            await ws.send_json({'type': 'error', 'msg': 'Invalid JSON'})
                    elif msg.type == WSMsgType.BINARY:
                        await self._handle_binary(ws, msg.data, client_state)
                    elif msg.type == WSMsgType.PING:
                        await ws.pong(msg.data)
                    elif msg.type == WSMsgType.PONG:
                        client_state['last_ping'] = time.time()
                    elif msg.type == WSMsgType.CLOSE:
                        break
            except (web.ConnectionClosed, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.error(f"WS reader error: {e}")
                log_error_to_file(f"WS reader: {e}")
            finally:
                client_state['closed'] = True

        async def writer():
            while not ws.closed:
                try:
                    msg = await asyncio.wait_for(send_queue.get(), timeout=1.0)
                    if ws.closed:
                        break
                    await ws.send_str(msg)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.warning(f"WS writer error: {e}")
                    break

        reader_task = asyncio.create_task(reader())
        writer_task = asyncio.create_task(writer())

        try:
            await asyncio.gather(reader_task, writer_task, return_exceptions=True)
        finally:
            self.websockets.discard(ws)
            for t in (reader_task, writer_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
            try:
                await ws.close()
            except Exception:
                pass

        return ws

    async def _handle_binary(self, ws, data: bytes, client_state):
        logger.info(f"Received {len(data)} bytes of binary data from {client_state['username']}")
        await ws.send_bytes(b"ACK_BINARY")

    async def _handle_ws_message(self, ws, data, client_state):
        msg_type = data.get('type')
        if msg_type == 'join':
            channel = data.get('channel', 'general')
            if channel in self.channels:
                client_state['channel'] = channel
                await self._send_state(ws, client_state)
        elif msg_type == 'set_username':
            client_state['username'] = data.get('username', 'human')[:20]
        elif msg_type == 'message':
            content = data.get('content', '')
            if content.startswith('/'):
                await self._handle_command(ws, content, client_state)
            else:
                store_id = client_state.get('channel', 'general')
                username = client_state.get('username', 'human')
                responses = []
                async for res in self.agent_loop.process_message(store_id, content, username, channel_name=store_id):
                    responses.append(res)
                # Save user message
                user_msg = {
                    'id': str(uuid.uuid4()),
                    'store_id': store_id,
                    'sender': username,
                    'sender_type': 'human',
                    'content': content,
                    'parent_id': '',
                    'thread_id': '',
                    'timestamp': int(time.time()*1000),
                    'reactions': {},
                    'pinned': 0
                }
                await db.execute(
                    'INSERT OR REPLACE INTO messages (id, store_id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (user_msg['id'], user_msg['store_id'], user_msg['sender'], user_msg['sender_type'],
                     user_msg['content'], user_msg['parent_id'], user_msg['thread_id'],
                     user_msg['timestamp'], json.dumps(user_msg['reactions']), user_msg['pinned'])
                )
                await self.broadcast({'type': 'new_message', 'channel': store_id, 'message': user_msg})

                for res in responses:
                    if res.get('reply'):
                        agent_msg = {
                            'id': str(uuid.uuid4()),
                            'store_id': store_id,
                            'sender': res.get('agent', 'Chef'),
                            'sender_type': 'agent',
                            'content': res['reply'],
                            'parent_id': '',
                            'thread_id': '',
                            'timestamp': int(time.time()*1000),
                            'reactions': {},
                            'pinned': 0
                        }
                        await db.execute(
                            'INSERT OR REPLACE INTO messages (id, store_id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (agent_msg['id'], agent_msg['store_id'], agent_msg['sender'], agent_msg['sender_type'],
                             agent_msg['content'], agent_msg['parent_id'], agent_msg['thread_id'],
                             agent_msg['timestamp'], json.dumps(agent_msg['reactions']), agent_msg['pinned'])
                        )
                        await self.broadcast({'type': 'new_message', 'channel': store_id, 'message': agent_msg})
                        await bio.heartbeat(content, res['reply'])

        elif msg_type == 'get_history':
            store_id = data.get('channel', 'general')
            rows = await db.fetchall('SELECT id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned FROM messages WHERE store_id=? ORDER BY timestamp DESC LIMIT 100', (store_id,))
            msgs = []
            for row in rows:
                msgs.append({
                    'id': row[0], 'sender': row[1], 'sender_type': row[2], 'content': row[3],
                    'parent_id': row[4], 'thread_id': row[5], 'timestamp': row[6],
                    'reactions': json.loads(row[7]) if row[7] else {},
                    'pinned': bool(row[8])
                })
            await ws.send_json({'type': 'history', 'channel': store_id, 'messages': msgs[::-1]})

        elif msg_type == 'spawn_agent':
            name = data.get('name', f"Agent-{str(uuid.uuid4())[:4]}")
            model = data.get('model', settings.default_generate_model)
            system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
            channels = data.get('channels', self.channels)
            aid = str(uuid.uuid4())[:8]
            agent = {
                'id': aid, 'name': name, 'model': model,
                'system_prompt': system_prompt, 'channels': channels,
                'strict_channel': None, 'status': 'online',
                'is_embed_operator': False, 'is_code_moderator': False
            }
            self.agents[aid] = agent
            memory = DecentMem(aid)
            memory.save()
            self.memories[aid] = memory
            self.agent_metrics[aid] = self._generate_initial_metrics()
            await db.execute(
                'INSERT OR REPLACE INTO agents (id, name, model, system_prompt, channels, strict_channel, status, is_embed_operator, is_code_moderator) VALUES (?,?,?,?,?,?,?,?,?)',
                (aid, name, model, system_prompt, json.dumps(channels), None, 'online', 0, 0)
            )
            self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, self)
            self.ralph_engine = RalphEngine(self, self.agent_loop)
            await self.broadcast_agents()
            await ws.send_json({'type': 'spawn_confirm', 'agent': agent})

        elif msg_type == 'update_agent':
            aid = data.get('id')
            if aid in self.agents:
                agent = self.agents[aid]
                agent['name'] = data.get('name', agent['name'])
                agent['model'] = data.get('model', agent['model'])
                agent['system_prompt'] = data.get('system_prompt', agent['system_prompt'])
                agent['channels'] = data.get('channels', agent['channels'])
                agent['strict_channel'] = data.get('strict_channel', agent['strict_channel'])
                await db.execute(
                    'UPDATE agents SET name=?, model=?, system_prompt=?, channels=?, strict_channel=? WHERE id=?',
                    (agent['name'], agent['model'], agent['system_prompt'], json.dumps(agent['channels']),
                     agent['strict_channel'], aid)
                )
                self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, self)
                self.ralph_engine = RalphEngine(self, self.agent_loop)
                await self.broadcast_agents()
            else:
                await ws.send_json({'type': 'error', 'msg': f'Agent {aid} not found'})

        elif msg_type == 'remove_agent':
            aid = data.get('id')
            if aid in self.agents and aid != 'moderator':
                del self.agents[aid]
                self.memories.pop(aid, None)
                self.agent_metrics.pop(aid, None)
                await db.execute('DELETE FROM agents WHERE id=?', (aid,))
                self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, self)
                self.ralph_engine = RalphEngine(self, self.agent_loop)
                await self.broadcast_agents()
            else:
                await ws.send_json({'type': 'error', 'msg': 'Cannot remove moderator or unknown agent'})

        elif msg_type == 'siphon_start':
            query = data.get('query', '')
            if query:
                sid = self.research_engine.start_siphon(query, 'siphon')
                await ws.send_json({'type': 'siphon_started', 'session_id': sid})

        elif msg_type == 'dataset_run':
            name = data.get('name')
            if name:
                try:
                    job_id = await self.dataset_engine.run_dataset_job(name)
                    await ws.send_json({'type': 'dataset_job_started', 'job_id': job_id})
                except Exception as e:
                    await ws.send_json({'type': 'error', 'msg': str(e)})

        elif msg_type == 'dataset_list':
            files = list(settings.datasets_dir.glob("*.md")) + list(settings.base_dir.glob("*.md"))
            names = [f.stem for f in files if "dataset" in f.name.lower() or "datasets" in f.name.lower()]
            await ws.send_json({'type': 'dataset_list', 'datasets': names})

        elif msg_type == 'reload_skills':
            await self.skill_engine.reload()
            await self.broadcast_skills()

        elif msg_type == 'request_metrics':
            await self.broadcast_metrics()

        elif msg_type == 'get_ralph_sessions':
            sessions = self.ralph_engine.get_sessions() if self.ralph_engine else []
            await ws.send_json({'type': 'ralph_sessions', 'sessions': sessions})

        elif msg_type == 'ralph_start':
            topic = data.get('topic')
            if topic:
                sid = self.ralph_engine.start_ralph(topic, 'ralph')
                await ws.send_json({'type': 'ralph_started', 'session_id': sid})
            else:
                await ws.send_json({'type': 'error', 'msg': 'No topic provided'})

        elif msg_type == 'abstract_start':
            query = data.get('query')
            if query:
                store_id = client_state.get('channel', 'general')
                await self.abstract_engine.generate_plan(query, store_id)
                await ws.send_json({'type': 'notification', 'msg': f"Abstract plan generated for: {query}"})
            else:
                await ws.send_json({'type': 'error', 'msg': 'No query provided'})

        elif msg_type == 'get_siphon_history':
            rows = await db.fetchall('SELECT content, timestamp FROM messages WHERE store_id="siphon" ORDER BY timestamp DESC LIMIT 50')
            history = [{'content': row[0], 'timestamp': row[1]} for row in rows]
            await ws.send_json({'type': 'siphon_history', 'history': history})

        elif msg_type == 'close':
            code = data.get('code', 1000)
            reason = data.get('reason', '')
            await ws.close(code=code, message=reason)

    async def _handle_command(self, ws, cmd, client_state):
        parts = shlex.split(cmd)
        if not parts:
            return
        command = parts[0].lower()
        args = parts[1:]

        if command == '/help':
            help_text = """
HUMBOLDT-CHEF Commands:
  /help            - This help
  /skills          - List loaded skills
  /agents          - List active agents
  /bio             - Show bio.md
  /soul            - Show soul.md
  /status          - Ollama status
  /errorlog        - Show error log
  /siphon <query>  - Start a full research (search, scrape, summarize)
  /dataset <name>  - Run dataset scraping
  /datasets        - List dataset files
  /ralph <topic>   - Start evolutionary brainstorming
  /abstract <query> - Generate a step‑by‑step plan
  /stack list      - List stack templates
  /stack build <name> - Build a repo from template
  /stack add <description> - Create a template from description
  /stack import <json> - Bulk import templates
  /spawn_interactive - Interactive agent spawn
  /update_agent <id> <field> <value> - Update agent
  /remove_agent <id> - Remove agent
  /reverse_skill <description> - Generate skill
  /quit            - Exit (CLI only)
"""
            await ws.send_json({'type': 'notification', 'msg': help_text})
        elif command == '/skills':
            skills = self.skill_engine.list()
            msg = "Skills:\n" + "\n".join([f"  {s['name']}: {s['description']}" for s in skills])
            await ws.send_json({'type': 'notification', 'msg': msg})
        elif command == '/agents':
            agent_list = [f"{aid}: {a['name']} ({a['model']})" for aid, a in self.agents.items()]
            await ws.send_json({'type': 'notification', 'msg': "Agents:\n" + "\n".join(agent_list)})
        elif command == '/bio':
            content = read_file(settings.bio_file)
            await ws.send_json({'type': 'notification', 'msg': f"BIO.md:\n{content}"})
        elif command == '/soul':
            content = read_file(settings.soul_file)
            await ws.send_json({'type': 'notification', 'msg': f"SOUL.md:\n{content}"})
        elif command == '/status':
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{settings.ollama_url}/api/tags", timeout=2) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            models = [m['name'] for m in data.get('models', [])]
                            await ws.send_json({'type': 'notification', 'msg': f"Ollama reachable. Models: {', '.join(models)}"})
                        else:
                            await ws.send_json({'type': 'notification', 'msg': "Ollama reachable but returned error."})
            except:
                await ws.send_json({'type': 'notification', 'msg': "Ollama is not reachable."})
        elif command == '/errorlog':
            content = read_file(settings.error_log_path)
            await ws.send_json({'type': 'notification', 'msg': f"Error Log:\n{content}"})
        elif command == '/siphon':
            if len(args) >= 1:
                query = ' '.join(args)
                sid = self.research_engine.start_siphon(query, 'siphon')
                await ws.send_json({'type': 'notification', 'msg': f"Research started: {sid}"})
            else:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /siphon <query>'})
        elif command == '/dataset':
            if len(args) >= 1:
                name = args[0]
                try:
                    job_id = await self.dataset_engine.run_dataset_job(name)
                    await ws.send_json({'type': 'notification', 'msg': f"Dataset job started: {job_id}"})
                except Exception as e:
                    await ws.send_json({'type': 'error', 'msg': str(e)})
            else:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /dataset <name>'})
        elif command == '/datasets':
            files = list(settings.datasets_dir.glob("*.md")) + list(settings.base_dir.glob("*.md"))
            names = [f.stem for f in files if "dataset" in f.name.lower() or "datasets" in f.name.lower()]
            if names:
                await ws.send_json({'type': 'notification', 'msg': f"Available datasets: {', '.join(names)}"})
            else:
                await ws.send_json({'type': 'notification', 'msg': "No dataset files found."})
        elif command == '/ralph':
            if len(args) >= 1:
                topic = ' '.join(args)
                sid = self.ralph_engine.start_ralph(topic, 'ralph')
                await ws.send_json({'type': 'notification', 'msg': f"Ralph session started: {sid}"})
            else:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /ralph <topic>'})
        elif command == '/abstract':
            if len(args) >= 1:
                query = ' '.join(args)
                store_id = client_state.get('channel', 'general')
                await self.abstract_engine.generate_plan(query, store_id)
                await ws.send_json({'type': 'notification', 'msg': f"Abstract plan generated for: {query}"})
            else:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /abstract <query>'})
        elif command == '/stack':
            if len(args) == 0:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /stack list|build|add|import'})
                return
            sub = args[0].lower()
            if sub == 'list':
                templates = list(self.stack_engine.templates.keys())
                await ws.send_json({'type': 'notification', 'msg': f"Templates: {', '.join(templates) if templates else 'None'}"})
            elif sub == 'build':
                if len(args) < 2:
                    await ws.send_json({'type': 'error', 'msg': 'Usage: /stack build <name>'})
                    return
                name = args[1]
                target_dir = settings.workspace_dir / f"stack_{name}_{int(time.time())}"
                success = await self.stack_engine.build(name, target_dir)
                if success:
                    await ws.send_json({'type': 'notification', 'msg': f"Template '{name}' built at {target_dir}"})
                else:
                    await ws.send_json({'type': 'error', 'msg': f"Template '{name}' not found"})
            elif sub == 'add':
                if len(args) < 2:
                    await ws.send_json({'type': 'error', 'msg': 'Usage: /stack add <description>'})
                    return
                desc = ' '.join(args[1:])
                result = await self.stack_engine.add(desc)
                await ws.send_json({'type': 'notification', 'msg': result})
            elif sub == 'import':
                if len(args) < 2:
                    await ws.send_json({'type': 'error', 'msg': 'Usage: /stack import <json>'})
                    return
                json_str = ' '.join(args[1:])
                count = await self.stack_engine.import_(json_str)
                if count >= 0:
                    await ws.send_json({'type': 'notification', 'msg': f"Imported {count} templates."})
                else:
                    await ws.send_json({'type': 'error', 'msg': "Invalid JSON"})
            else:
                await ws.send_json({'type': 'error', 'msg': f"Unknown stack subcommand: {sub}"})
        elif command == '/update_agent':
            if len(args) >= 3:
                aid = args[0]
                field = args[1]
                value = ' '.join(args[2:])
                if aid in self.agents:
                    agent = self.agents[aid]
                    if field == 'name':
                        agent['name'] = value.strip('"')
                    elif field == 'model':
                        agent['model'] = value
                    elif field == 'system_prompt':
                        agent['system_prompt'] = value.strip('"')
                    elif field == 'channels':
                        agent['channels'] = [c.strip() for c in value.split(',')]
                    else:
                        await ws.send_json({'type': 'error', 'msg': f'Unknown field: {field}'})
                        return
                    await db.execute(
                        'UPDATE agents SET name=?, model=?, system_prompt=?, channels=? WHERE id=?',
                        (agent['name'], agent['model'], agent['system_prompt'], json.dumps(agent['channels']), aid)
                    )
                    self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, self)
                    self.ralph_engine = RalphEngine(self, self.agent_loop)
                    await self.broadcast_agents()
                    await ws.send_json({'type': 'notification', 'msg': f'Agent {aid} updated'})
                else:
                    await ws.send_json({'type': 'error', 'msg': f'Agent {aid} not found'})
            else:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /update_agent <id> <field> <value>'})
        elif command == '/spawn_interactive':
            await ws.send_json({'type': 'notification', 'msg': 'Use the "Spawn Agent" button in the UI.'})
        elif command == '/remove_agent':
            if len(args) >= 1:
                aid = args[0]
                if aid in self.agents and aid != 'moderator':
                    del self.agents[aid]
                    self.memories.pop(aid, None)
                    await db.execute('DELETE FROM agents WHERE id=?', (aid,))
                    self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, self)
                    self.ralph_engine = RalphEngine(self, self.agent_loop)
                    await self.broadcast_agents()
                    await ws.send_json({'type': 'notification', 'msg': f"Agent {aid} removed."})
                else:
                    await ws.send_json({'type': 'error', 'msg': "Agent not found or cannot remove."})
            else:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /remove_agent <id>'})
        elif command == '/reverse_skill':
            if len(args) >= 1:
                description = ' '.join(args)
                result = await self.skill_engine.reverse_skill(description)
                await ws.send_json({'type': 'notification', 'msg': json.dumps(result, indent=2)})
            else:
                await ws.send_json({'type': 'error', 'msg': 'Usage: /reverse_skill <description>'})
        elif command == '/graph':
            await ws.send_json({'type': 'error', 'msg': 'Graph feature has been removed.'})
        else:
            await ws.send_json({'type': 'error', 'msg': f'Unknown command: {command}'})

    async def _send_state(self, ws, client_state):
        channel = client_state.get('channel', 'general')
        await ws.send_json({'type': 'channels', 'channels': self.channels})
        await self.broadcast_agents()
        await self.broadcast_skills()
        rows = await db.fetchall('SELECT id, sender, sender_type, content, parent_id, thread_id, timestamp, reactions, pinned FROM messages WHERE store_id=? ORDER BY timestamp DESC LIMIT 100', (channel,))
        msgs = []
        for row in rows:
            msgs.append({
                'id': row[0], 'sender': row[1], 'sender_type': row[2], 'content': row[3],
                'parent_id': row[4], 'thread_id': row[5], 'timestamp': row[6],
                'reactions': json.loads(row[7]) if row[7] else {},
                'pinned': bool(row[8])
            })
        await ws.send_json({'type': 'history', 'channel': channel, 'messages': msgs[::-1]})

    async def broadcast(self, message):
        dead = set()
        payload = json.dumps(message, ensure_ascii=False)
        for ws in list(self.websockets):
            if ws.closed:
                dead.add(ws)
                continue
            try:
                if hasattr(ws, '_send_queue') and ws._send_queue:
                    try:
                        await asyncio.wait_for(ws._send_queue.put(payload), timeout=0.5)
                    except asyncio.TimeoutError:
                        dead.add(ws)
                else:
                    await asyncio.wait_for(ws.send_str(payload), timeout=1.0)
            except (asyncio.TimeoutError, web.ConnectionClosed, RuntimeError):
                dead.add(ws)
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")
                dead.add(ws)

        for ws in dead:
            self.websockets.discard(ws)
            try:
                await ws.close()
            except Exception:
                pass

    async def broadcast_agents(self):
        agent_list = [{'id': aid, **agent} for aid, agent in self.agents.items()]
        await self.broadcast({'type': 'agents_list', 'agents': agent_list})

    async def broadcast_skills(self):
        skills = self.skill_engine.list()
        await self.broadcast({'type': 'skills_list', 'skills': skills})

    async def handle_metrics(self, request):
        return web.json_response({'agents': self.agent_metrics})

    # Enhanced handle_models with caching and force refresh
    async def handle_models(self, request):
        # Check if refresh is requested
        refresh = request.query.get('refresh', '').lower() == 'true'
        now = time.time()

        # Return cached models if fresh (5s TTL) and not forced refresh
        if not refresh and (now - self._models_cache_time < 5) and self._cached_models:
            return web.json_response({'models': self._cached_models})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.ollama_url}/api/tags", timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._cached_models = [m['name'] for m in data.get('models', [])]
                        self._models_cache_time = now
                        logger.info(f"Ollama models fetched (cached): {self._cached_models}")
                        return web.json_response({'models': self._cached_models})
                    else:
                        logger.warning(f"Ollama /api/tags returned status {resp.status}")
        except asyncio.TimeoutError:
            logger.error("Ollama /api/tags timed out after 10s")
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching models: {e}")

        # Fallback: return cached models if any, else default
        if not self._cached_models:
            self._cached_models = [settings.default_generate_model]
        return web.json_response({'models': self._cached_models})

    async def handle_errorlog(self, request):
        content = read_file(settings.error_log_path)
        return web.json_response({'log': content})

    async def handle_health(self, request):
        ollama_healthy = False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.ollama_url}/api/tags", timeout=2) as resp:
                    ollama_healthy = resp.status == 200
        except:
            pass

        return web.json_response({
            'status': 'healthy' if ollama_healthy else 'degraded',
            'ollama': ollama_healthy,
            'agents': len(self.agents),
            'websockets': len(self.websockets),
            'skills': len(self.skill_engine.registry),
            'templates': len(self.stack_engine.templates),
            'timestamp': time.time()
        })

    async def handle_heartbeat(self, request):
        return web.json_response({'status': 'ok'})

    def shutdown(self):
        self._shutdown_event.set()

# ==============================================================================
# SECTION 17: CLI MODE
# PURPOSE: Runs the agent system without the web UI, accepting commands from
#          the terminal. Useful for headless operation or testing.
# EXTENSION: Add more interactive commands.
# ==============================================================================
class CLIMode:
    def __init__(self):
        self.skill_engine = SkillEngine()
        self.dataset_engine = DatasetEngine(None)
        self.moderator = Moderator()
        self.stack_engine = StackEngine()
        self.abstract_engine = AbstractEngine(None)
        self.agent_loop = None
        self.ralph_engine = None
        self.agents = {}
        self.memories = {}

    async def run(self):
        await db.init()
        print(colorize("HUMBOLDT-CHEF CLI Mode (type /help for commands)", "cyan"))
        await self.skill_engine.reload()
        await self.stack_engine.load_templates()
        cli_agent_id = 'cli_agent'
        self.agents[cli_agent_id] = {
            'id': cli_agent_id,
            'name': 'CLI Chef',
            'model': settings.default_generate_model,
            'system_prompt': 'You are a helpful assistant.',
            'channels': ['general'],
            'status': 'online',
            'is_embed_operator': False,
            'is_code_moderator': False
        }
        self.memories[cli_agent_id] = DecentMem(cli_agent_id)
        self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, None)
        self.ralph_engine = RalphEngine(None, self.agent_loop)

        while True:
            try:
                user_input = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
            except (asyncio.CancelledError, KeyboardInterrupt):
                break
            if not user_input:
                break
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith('/'):
                await self._handle_cli_command(user_input)
            else:
                async for res in self.agent_loop.process_message('general', user_input, 'user', channel_name='general'):
                    response = res.get('reply', '')
                    print(colorize(f"[Chef] {response}", "green"))
                    await bio.heartbeat(user_input, response)

    async def _handle_cli_command(self, cmd):
        parts = shlex.split(cmd)
        if not parts:
            return
        command = parts[0].lower()
        args = parts[1:]

        if command == '/help':
            print(colorize("""
HUMBOLDT-CHEF CLI Commands:
  /help            - This help
  /skills          - List loaded skills
  /bio             - Show bio.md
  /soul            - Show soul.md
  /status          - Ollama status
  /errorlog        - Show error log
  /siphon <query>  - Start a full research (search, scrape, summarize)
  /dataset <name>  - Run dataset scraping
  /datasets        - List dataset files
  /ralph <topic>   - Start evolutionary brainstorming
  /abstract <query> - Generate a step‑by‑step plan
  /stack list      - List stack templates
  /stack build <name> - Build a repo from template
  /stack add <description> - Create a template from description
  /stack import <json> - Bulk import templates
  /spawn_interactive - Interactive agent spawn
  /update_agent <id> <field> <value> - Update agent
  /remove_agent <id> - Remove agent
  /reverse_skill <description> - Generate skill
  /quit            - Exit
""", "green"))
        elif command == '/skills':
            skills = self.skill_engine.list()
            if skills:
                print(colorize("Skills:", "cyan"))
                for s in skills:
                    print(f"  - {s['name']}: {s['description']}")
            else:
                print(colorize("No skills loaded.", "yellow"))
        elif command == '/bio':
            content = read_file(settings.bio_file)
            print(colorize(f"BIO.md:\n{content}", "blue"))
        elif command == '/soul':
            content = read_file(settings.soul_file)
            print(colorize(f"SOUL.md:\n{content}", "blue"))
        elif command == '/status':
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{settings.ollama_url}/api/tags", timeout=2) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            models = [m['name'] for m in data.get('models', [])]
                            print(colorize(f"Ollama reachable. Models: {', '.join(models)}", "yellow"))
                        else:
                            print(colorize("Ollama reachable but returned error.", "red"))
            except:
                print(colorize("Ollama is not reachable.", "red"))
        elif command == '/errorlog':
            content = read_file(settings.error_log_path)
            print(colorize(f"Error Log:\n{content}", "red"))
        elif command == '/siphon':
            if len(args) >= 1:
                query = ' '.join(args)
                research = ResearchEngine(None)
                sid = research.start_siphon(query, 'siphon')
                print(colorize(f"Research started: {sid}. Check logs.", "cyan"))
            else:
                print(colorize("Usage: /siphon <query>", "yellow"))
        elif command == '/dataset':
            if len(args) >= 1:
                name = args[0]
                try:
                    job_id = await self.dataset_engine.run_dataset_job(name)
                    print(colorize(f"Dataset job started: {job_id}", "green"))
                except Exception as e:
                    print(colorize(f"Dataset error: {e}", "red"))
            else:
                print(colorize("Usage: /dataset <name>", "yellow"))
        elif command == '/datasets':
            files = list(settings.datasets_dir.glob("*.md")) + list(settings.base_dir.glob("*.md"))
            names = [f.stem for f in files if "dataset" in f.name.lower() or "datasets" in f.name.lower()]
            if names:
                print(colorize(f"Available datasets: {', '.join(names)}", "cyan"))
            else:
                print(colorize("No dataset files found.", "yellow"))
        elif command == '/ralph':
            if len(args) >= 1:
                topic = ' '.join(args)
                sid = self.ralph_engine.start_ralph(topic, 'ralph')
                print(colorize(f"Ralph session started: {sid}", "green"))
            else:
                print(colorize("Usage: /ralph <topic>", "yellow"))
        elif command == '/abstract':
            if len(args) >= 1:
                query = ' '.join(args)
                await self.abstract_engine.generate_plan(query, 'general')
                print(colorize(f"Abstract plan generated for: {query}", "green"))
            else:
                print(colorize("Usage: /abstract <query>", "yellow"))
        elif command == '/stack':
            if len(args) == 0:
                print(colorize("Usage: /stack list|build|add|import", "yellow"))
                return
            sub = args[0].lower()
            if sub == 'list':
                templates = list(self.stack_engine.templates.keys())
                print(colorize(f"Templates: {', '.join(templates) if templates else 'None'}", "cyan"))
            elif sub == 'build':
                if len(args) < 2:
                    print(colorize("Usage: /stack build <name>", "yellow"))
                    return
                name = args[1]
                target_dir = settings.workspace_dir / f"stack_{name}_{int(time.time())}"
                success = await self.stack_engine.build(name, target_dir)
                if success:
                    print(colorize(f"Template '{name}' built at {target_dir}", "green"))
                else:
                    print(colorize(f"Template '{name}' not found", "red"))
            elif sub == 'add':
                if len(args) < 2:
                    print(colorize("Usage: /stack add <description>", "yellow"))
                    return
                desc = ' '.join(args[1:])
                result = await self.stack_engine.add(desc)
                print(colorize(result, "cyan"))
            elif sub == 'import':
                if len(args) < 2:
                    print(colorize("Usage: /stack import <json>", "yellow"))
                    return
                json_str = ' '.join(args[1:])
                count = await self.stack_engine.import_(json_str)
                if count >= 0:
                    print(colorize(f"Imported {count} templates.", "green"))
                else:
                    print(colorize("Invalid JSON", "red"))
            else:
                print(colorize(f"Unknown stack subcommand: {sub}", "red"))
        elif command == '/spawn_interactive':
            name = input("Agent name: ").strip() or f"Agent-{str(uuid.uuid4())[:4]}"
            model = input(f"Model ({settings.default_generate_model}): ").strip() or settings.default_generate_model
            system_prompt = input("System prompt (or press Enter for default): ").strip() or "You are a helpful assistant."
            channels_input = input("Channels (comma-separated, e.g., general,code): ").strip()
            channels = [c.strip() for c in channels_input.split(',')] if channels_input else ['general']

            aid = str(uuid.uuid4())[:8]
            agent = {
                'id': aid, 'name': name, 'model': model,
                'system_prompt': system_prompt, 'channels': channels,
                'status': 'online', 'is_embed_operator': False, 'is_code_moderator': False
            }
            self.agents[aid] = agent
            self.memories[aid] = DecentMem(aid)
            self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, None)
            self.ralph_engine = RalphEngine(None, self.agent_loop)
            print(colorize(f"Agent {name} spawned with ID: {aid}", "green"))
        elif command == '/update_agent':
            if len(args) >= 3:
                aid = args[0]
                field = args[1]
                value = ' '.join(args[2:])
                if aid in self.agents:
                    agent = self.agents[aid]
                    if field == 'name':
                        agent['name'] = value.strip('"')
                    elif field == 'model':
                        agent['model'] = value
                    elif field == 'system_prompt':
                        agent['system_prompt'] = value.strip('"')
                    elif field == 'channels':
                        agent['channels'] = [c.strip() for c in value.split(',')]
                    else:
                        print(colorize(f"Unknown field: {field}", "red"))
                        return
                    self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, None)
                    self.ralph_engine = RalphEngine(None, self.agent_loop)
                    print(colorize(f"Agent {aid} updated", "green"))
                else:
                    print(colorize(f"Agent {aid} not found", "red"))
            else:
                print(colorize("Usage: /update_agent <id> <field> <value>", "yellow"))
        elif command == '/remove_agent':
            if len(args) >= 1:
                aid = args[0]
                if aid in self.agents:
                    del self.agents[aid]
                    self.memories.pop(aid, None)
                    self.agent_loop = AgentLoop(self.agents, self.memories, self.skill_engine, self.moderator, None)
                    self.ralph_engine = RalphEngine(None, self.agent_loop)
                    print(colorize(f"Agent {aid} removed", "green"))
                else:
                    print(colorize(f"Agent {aid} not found", "red"))
            else:
                print(colorize("Usage: /remove_agent <id>", "yellow"))
        elif command == '/reverse_skill':
            if len(args) >= 1:
                description = ' '.join(args)
                result = await self.skill_engine.reverse_skill(description)
                print(colorize(json.dumps(result, indent=2), "cyan"))
            else:
                print(colorize("Usage: /reverse_skill <description>", "yellow"))
        elif command == '/quit':
            sys.exit(0)
        else:
            print(colorize(f"Unknown command: {command}. Type /help.", "red"))

# ==============================================================================
# SECTION 18: INDEX HTML (Web UI)
# PURPOSE: The complete single‑page application served at the root.
#          Contains the chat interface, agent management, and all tabs.
#          FIXED: initial fetch and retry both force server refresh.
# EXTENSION: Modify UI elements, add new tabs, or improve styling.
# ==============================================================================
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <meta name="description" content="OPENCHEF! – Skill-First Multi-Agent Harness">
    <meta name="theme-color" content="#0f1a1a">
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%3C/text%3E%3C/svg%3E">
    <title>OPENCHEF! — Skill Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        :root { --bg-dark: #000000; --bg-darker: #0a0a0a; --bg-card: #111111; --bg-surface: #1a1a1a; --primary: #ff6b35; --primary-glow: rgba(255,107,53,0.25); --secondary: #00d4aa; --accent: #4d8fff; --warning: #ffcc00; --error: #ff3366; --success: #00ff88; --text-primary: #ffffff; --text-secondary: #cccccc; --text-dim: #888888; --border: #222222; --border-light: #2a2a2a; --font-mono: 'Courier New', 'Fira Code', 'Consolas', monospace; }
        body { font-family: var(--font-mono); background: var(--bg-dark); color: var(--text-primary); height: 100vh; overflow: hidden; font-size: 13px; }
        .terminal { display: flex; flex-direction: column; height: 100vh; }
        .header { background: var(--bg-darker); border-bottom: 2px solid var(--primary); padding: 6px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.5); display: flex; justify-content: space-between; align-items: center; }
        .header-left { display: flex; flex-direction: column; }
        .header-right { display: flex; gap: 20px; align-items: center; font-size: 11px; color: var(--text-secondary); }
        .header-right span { background: var(--bg-surface); padding: 2px 10px; border-radius: 12px; border: 1px solid var(--border-light); }
        .ascii-art { color: var(--primary); font-size: 6.2px; line-height: 1.05; white-space: pre; font-family: monospace; letter-spacing: 0.3px; }
        .subtitle { color: var(--secondary); font-size: 9px; margin-top: 3px; letter-spacing: 1.5px; text-transform: uppercase; }
        .container { display: flex; flex: 1; overflow: hidden; gap: 1px; background: var(--border); }
        .cli-panel { flex: 1.5; display: flex; flex-direction: column; background: var(--bg-dark); }
        .cli-output { flex: 1; padding: 14px 16px; overflow-y: auto; background: var(--bg-darker); font-family: var(--font-mono); font-size: 12px; }
        .cli-line { margin-bottom: 5px; padding-left: 22px; position: relative; word-wrap: break-word; white-space: pre-wrap; animation: fadeIn 0.12s ease-out; border-left: 1px solid transparent; }
        .cli-prompt { color: var(--primary); font-weight: bold; position: absolute; left: 0; top: 0; }
        .cli-command { color: var(--primary); font-weight: 500; }
        .cli-success { color: var(--success); text-shadow: 0 0 2px rgba(0,255,136,0.3); }
        .cli-error { color: var(--error); }
        .cli-warning { color: var(--warning); }
        .cli-info { color: var(--secondary); }
        .cli-question { color: var(--warning); font-weight: bold; }
        .cli-link { color: var(--accent); text-decoration: none; cursor: pointer; border-bottom: 1px dashed var(--accent); }
        .cli-link:hover { color: var(--primary); border-bottom-color: var(--primary); }
        .cli-input-container { background: var(--bg-card); border-top: 1px solid var(--border-light); padding: 10px 16px; display: flex; gap: 10px; align-items: center; }
        .cli-prompt-sign { color: var(--primary); font-weight: bold; font-size: 14px; background: rgba(255,107,53,0.15); padding: 2px 8px; border-radius: 20px; }
        .cli-input { flex: 1; background: transparent; border: none; color: var(--text-primary); font-family: var(--font-mono); font-size: 13px; outline: none; padding: 6px 0; }
        .cli-input::placeholder { color: var(--text-dim); font-style: italic; }
        .sidebar { width: 480px; background: var(--bg-card); display: flex; flex-direction: column; overflow: hidden; border-left: 1px solid var(--border); }
        .sidebar-tabs { display: flex; background: var(--bg-darker); border-bottom: 1px solid var(--border); gap: 2px; padding: 0 12px; }
        .tab-btn { background: transparent; border: none; color: var(--text-dim); padding: 10px 18px; font-family: var(--font-mono); font-size: 11px; font-weight: bold; cursor: pointer; transition: all 0.15s; letter-spacing: 1px; border-bottom: 2px solid transparent; }
        .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); background: rgba(255,107,53,0.05); }
        .tab-btn:hover:not(.active) { color: var(--secondary); background: rgba(0,212,170,0.05); }
        .tab-content { flex: 1; overflow-y: auto; padding: 12px; display: none; }
        .tab-content.active { display: block; }
        .ref-two-columns { display: flex; gap: 16px; height: 100%; }
        .ref-left { flex: 1; overflow-y: auto; padding-right: 8px; border-right: 1px solid var(--border); }
        .ref-right { flex: 1.2; overflow-y: auto; }
        .skill-list-compact { display: flex; flex-direction: column; gap: 8px; }
        .skill-item { background: var(--bg-surface); padding: 8px 10px; border-radius: 8px; cursor: pointer; font-size: 11px; display: flex; justify-content: space-between; align-items: center; border: 1px solid var(--border); transition: all 0.1s ease; }
        .skill-item.active { border-left: 3px solid var(--primary); background: rgba(255,107,53,0.1); }
        .skill-item:hover { background: rgba(255,107,53,0.08); border-color: var(--primary-glow); }
        .skill-info { flex: 1; }
        .skill-name { font-weight: bold; color: var(--secondary); font-size: 11px; display: flex; gap: 5px; align-items: center; }
        .skill-meta { font-size: 9px; color: var(--text-dim); margin-top: 3px; }
        .icon-btn { background: rgba(255,107,53,0.12); border: none; color: var(--primary); padding: 4px 8px; border-radius: 6px; cursor: pointer; font-size: 9px; font-weight: bold; transition: 0.07s linear; }
        .icon-btn:hover { background: var(--primary); color: black; transform: scale(0.96); }
        .caps-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; margin-bottom: 16px; }
        .cap-card { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 20px; padding: 6px 12px; font-size: 10px; cursor: pointer; transition: 0.05s linear; color: var(--text-secondary); }
        .cap-card:hover { background: var(--accent); color: black; border-color: var(--accent); transform: scale(1.02); }
        .category-title { color: var(--accent); font-size: 11px; margin: 12px 0 6px 0; font-weight: bold; border-left: 3px solid var(--primary); padding-left: 8px; }
        .quick-cmd-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
        .quick-cmd { background: rgba(77,143,255,0.12); border: 1px solid var(--accent); border-radius: 20px; padding: 4px 12px; font-size: 9px; cursor: pointer; font-family: monospace; }
        .quick-cmd:hover { background: var(--accent); color: black; }
        .ref-guide-box { background: var(--bg-surface); border-radius: 12px; padding: 12px; margin-top: 16px; border-left: 3px solid var(--secondary); }
        .guide-title { color: var(--secondary); font-weight: bold; font-size: 11px; margin-bottom: 8px; }
        .stats-panel { background: var(--bg-surface); border-radius: 12px; padding: 10px; margin-top: 12px; }
        .footer { background: var(--bg-darker); border-top: 1px solid var(--border); padding: 5px 16px; display: flex; justify-content: space-between; font-size: 9px; color: var(--text-dim); }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; margin-right: 6px; }
        .status-dot.online { background: var(--success); }
        .status-dot.working { background: var(--warning); }
        .status-dot.offline { background: var(--error); }
        .status-dot.ready { background: var(--success); animation: pulse 1.8s infinite; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-2px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-surface); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        .view-modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.96); z-index: 2000; display: none; flex-direction: column; }
        .view-modal.active { display: flex; }
        .view-header { background: var(--bg-darker); border-bottom: 1px solid var(--primary); padding: 12px 20px; display: flex; justify-content: space-between; }
        .view-title { color: var(--primary); font-weight: bold; }
        .view-close { background: rgba(255,107,53,0.2); border: 1px solid var(--primary); color: var(--primary); padding: 4px 16px; cursor: pointer; border-radius: 6px; }
        .view-content { flex: 1; overflow-y: auto; padding: 20px; font-family: monospace; font-size: 11px; white-space: pre-wrap; }
        #graphContainer { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 50000; justify-content: center; align-items: center; }
        #graphContainer .graph-wrap { background: var(--bg-dark); width: 90%; height: 90%; border: 2px solid var(--primary); position: relative; border-radius: 8px; }
        #graphContainer .close-graph { position: absolute; top: 10px; right: 20px; font-size: 28px; cursor: pointer; background: var(--primary); color: #000; padding: 0 12px; border-radius: 4px; z-index: 50001; }
        #graphNetwork { width: 100%; height: 100%; }
        .action-btn { background: rgba(255,107,53,0.15); border: 1px solid var(--primary); color: var(--primary); padding: 4px 12px; border-radius: 12px; cursor: pointer; font-size: 9px; font-weight: bold; margin: 2px; transition: 0.05s linear; }
        .action-btn:hover { background: var(--primary); color: black; }
        .progress-bar-container {
            width: 100%;
            height: 6px;
            background: var(--bg-surface);
            border-radius: 3px;
            margin: 4px 0;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: var(--secondary);
            transition: width 0.3s ease;
            border-radius: 3px;
        }
        .progress-bar-fill.complete {
            background: var(--success);
        }
        .progress-bar-fill.stuck {
            background: var(--warning);
        }
        #agentPopup {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--bg-card);
            border: 2px solid var(--primary);
            border-radius: 12px;
            padding: 24px;
            width: 400px;
            max-width: 90%;
            z-index: 9999;
            box-shadow: 0 8px 32px rgba(0,0,0,0.8);
        }
        #agentPopup input, #agentPopup select, #agentPopup textarea {
            width: 100%;
            background: var(--bg-surface);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px;
            border-radius: 6px;
            font-family: monospace;
        }
        #agentPopup textarea { resize: vertical; }
        #agentPopup .popup-actions { display: flex; gap: 8px; margin-top: 12px; }
        #agentPopup .popup-actions button { flex: 1; padding: 8px; border-radius: 6px; font-weight: bold; cursor: pointer; }
        #agentPopup .popup-actions .save { background: var(--primary); border: none; color: #000; }
        #agentPopup .popup-actions .cancel { background: var(--bg-surface); border: 1px solid var(--border); color: var(--text-primary); }
        #agentPopup .delete-btn { background: var(--error); border: none; color: #000; padding: 6px 12px; border-radius: 6px; font-weight: bold; cursor: pointer; margin-top: 12px; }
        #manualModelContainer { display: none; margin-top: 8px; }
        #manualModelContainer label { display: block; font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
        #manualModelContainer input { width: 100%; background: var(--bg-surface); border: 1px solid var(--border); color: var(--text-primary); padding: 8px; border-radius: 6px; font-family: monospace; }
        #manualModelContainer button { margin-top: 4px; }
        #modelLoading { display: none; color: var(--text-dim); font-size: 10px; margin-top: 4px; }
        #retryModelsBtn { display: none; background: var(--accent); border: none; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 9px; cursor: pointer; }
        #refreshModelsBtn { background: var(--secondary); border: none; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 9px; cursor: pointer; margin-top: 4px; }
        #ralphProgressDisplay { font-size: 10px; color: var(--text-dim); margin-top: 4px; }
        .agent-loading-bar {
            height: 2px;
            background: var(--secondary);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 2px;
            margin-top: 3px;
        }
        .global-metrics span {
            background: var(--bg-surface);
            padding: 2px 10px;
            border-radius: 12px;
            border: 1px solid var(--border-light);
            font-size: 10px;
        }
        .global-metrics .metric-value {
            color: var(--secondary);
            font-weight: bold;
        }
    </style>
</head>
<body>
<div class="terminal">
    <div class="header">
        <div class="header-left">
            <div class="ascii-art">
    ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███████╗███████╗██╗      
   ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██╔════╝██╔════╝██║      
   ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║█████╗  █████╗  ██║      
   ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██╔══██║██╔══╝  ██╔══╝  ╚═╝      
   ╚██████╔╝██║     ███████╗██║ ╚████║╚██████╗██║  ██║███████╗██║     ██╗      
    ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝      
                     SKILL-FIRST MULTI-AGENT HARNESS                          
            </div>
            <div class="subtitle">Ollama Models | Skill first Agent Factory</div>
        </div>
        <div class="header-right global-metrics">
            <span>CPU: <span id="metricCpu" class="metric-value">0</span>%</span>
            <span>RAM: <span id="metricMem" class="metric-value">0</span>%</span>
            <span>Tokens/s: <span id="metricToken" class="metric-value">0</span></span>
        </div>
    </div>
    <div class="container">
        <div class="cli-panel">
            <div class="cli-output" id="cliOutput">
                <div class="cli-line"><span class="cli-prompt">$</span><span class="cli-command"> humboldt-chef --skills-first</span></div>
                <div class="cli-success">HUMBOLDT-CHEF initialized with agents + control plane</div>
                <div class="cli-info">Skills: <span id="skillCountHeader">0</span> loaded</div>
                <div class="cli-info"> <span class="cli-link" data-cmd="/help">/help</span> | <span class="cli-link" data-cmd="/skills">/skills</span> | <span class="cli-link" data-cmd="/agents">/agents</span></div>
                <div class="cli-line"><span class="cli-prompt">$</span></div>
            </div>
            <div class="cli-input-container">
                <span class="cli-prompt-sign">></span>
                <input type="text" class="cli-input" id="cliInput" placeholder="Type /help for commands, or click any capability..." autofocus>
            </div>
        </div>
        <div class="sidebar">
            <div class="sidebar-tabs">
                <button class="tab-btn active" data-tab="agents">AGENTS</button>
                <button class="tab-btn" data-tab="skills">SKILLS</button>
                <button class="tab-btn" data-tab="ralph">RALPH</button>
                <button class="tab-btn" data-tab="siphon">SIPHON</button>
            </div>
            <!-- AGENTS TAB -->
            <div class="tab-content active" id="agentsTab">
                <div id="agentListContainer"></div>
                <div style="margin-top:12px;">
                    <button id="spawnAgentBtn" class="icon-btn" style="width:100%;">SPAWN AGENT</button>
                </div>
            </div>
            <!-- SKILLS TAB -->
            <div class="tab-content" id="skillsTab">
                <div class="ref-two-columns">
                    <div class="ref-left">
                        <div style="margin-bottom:8px; font-weight:bold; color:var(--secondary);">SKILLS</div>
                        <div id="combinedSkillList" class="skill-list-compact"></div>
                        <div style="margin-top:12px; display:flex; gap:8px;">
                            <button id="reloadSkillsBtn" class="icon-btn" style="flex:1;">RELOAD</button>
                            <button id="createSkillBtn" class="icon-btn" style="flex:1;">CREATE</button>
                        </div>
                    </div>
                    <div class="ref-right">
                        <div style="font-weight:bold; color:var(--accent);">QUICK ACTIONS</div>
                        <div id="combinedQuickActions" class="quick-cmd-row"></div>
                        <div class="ref-guide-box">
                            <div class="guide-title">QUICK REFERENCE</div>
                            <div style="font-size:10px;">
                                <span style="color:var(--primary);">/create_skill</span> create a new skill<br>
                                <span style="color:var(--primary);">/reverse_skill</span> generate from description<br>
                                <span style="color:var(--secondary);">/ralph</span> evolutionary brainstorming<br>
                                <span style="color:var(--accent);">/abstract</span> step‑by‑step plan<br>
                                <span style="color:var(--accent);">/stack</span> template management
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <!-- RALPH TAB -->
            <div class="tab-content" id="ralphTab">
                <div style="margin-bottom:12px;">
                    <button id="newRalphBtn" class="icon-btn">START NEW RALPH</button>
                </div>
                <div style="margin-bottom:12px; font-weight:bold; color:var(--secondary);">ACTIVE RALPH SESSIONS</div>
                <div id="ralphSessionsContainer"></div>
                <div style="margin-top:16px; background:var(--bg-surface); padding:12px; border-radius:8px; border-left:3px solid var(--warning);">
                    <div style="font-weight:bold; color:var(--warning);">Convergence & Stagnation</div>
                    <div id="ralphProgressDisplay" style="font-size:10px; color:var(--text-dim);">No active sessions</div>
                </div>
            </div>
            <!-- SIPHON TAB -->
            <div class="tab-content" id="siphonTab">
                <div style="margin-bottom:12px;">
                    <button id="newSiphonBtn" class="icon-btn">START NEW SIPHON</button>
                </div>
                <div style="font-weight:bold; color:var(--accent); margin-top:12px;">PAST RESEARCH</div>
                <div id="siphonHistoryList" style="font-size:10px; max-height:400px; overflow-y:auto;"></div>
            </div>
        </div>
    </div>
    <div class="footer"><div><span class="status-dot ready"></span> SEAMLESS MODE</div><div>Clickable CLI | Skills Engine</div><div id="skillCountFooter">0 skills</div></div>
</div>

<!-- Agent Popup -->
<div id="agentPopup">
    <div style="display:flex;justify-content:space-between;margin-bottom:16px;">
        <span style="font-size:18px;font-weight:bold;color:var(--primary);">Agent Settings</span>
        <button onclick="closeAgentPopup()" style="background:transparent;border:none;color:var(--text-dim);font-size:20px;cursor:pointer;">X</button>
    </div>
    <div id="agentPopupContent">
        <div style="margin-bottom:12px;">
            <label style="display:block;font-size:11px;color:var(--text-dim);margin-bottom:4px;">Name</label>
            <input id="popupName" type="text">
        </div>
        <div style="margin-bottom:12px;">
            <label style="display:block;font-size:11px;color:var(--text-dim);margin-bottom:4px;">Model</label>
            <select id="popupModel"></select>
            <button id="refreshModelsBtn" style="background:var(--secondary);border:none;color:#000;padding:2px 8px;border-radius:4px;font-size:9px;cursor:pointer;margin-top:4px;">⟳ Refresh</button>
            <div id="modelLoading">Loading models...</div>
            <button id="retryModelsBtn">Retry</button>
            <div id="manualModelContainer">
                <label>Or type model name:</label>
                <input id="manualModelInput" type="text" placeholder="e.g., llama3.2:3b">
                <button onclick="useManualModel()" class="icon-btn" style="margin-top:4px;">Use Manual</button>
            </div>
        </div>
        <div style="margin-bottom:12px;">
            <label style="display:block;font-size:11px;color:var(--text-dim);margin-bottom:4px;">System Prompt</label>
            <textarea id="popupSystemPrompt" rows="3"></textarea>
        </div>
        <div style="margin-bottom:16px;">
            <label style="display:block;font-size:11px;color:var(--text-dim);margin-bottom:4px;">Channels</label>
            <input id="popupChannels" type="text" placeholder="general,code">
        </div>
        <div class="popup-actions">
            <button class="save" onclick="saveAgentSettings()">SAVE</button>
            <button class="cancel" onclick="closeAgentPopup()">CANCEL</button>
        </div>
        <button class="delete-btn" onclick="deleteSelectedAgent()">DELETE AGENT</button>
    </div>
</div>

<div id="viewModal" class="view-modal"><div class="view-header"><span class="view-title" id="viewModalTitle">SKILL PREVIEW</span><button class="view-close" id="closeViewBtn">CLOSE</button></div><div class="view-content" id="viewModalContent"></div></div>

<script>
    const WS_PORT = {{WS_PORT}};
    // FIX: define MAX_GENERATIONS constant for Ralph progress
    const MAX_GENERATIONS = 30;   // match settings.max_generations from Python
    let ws, currentChannel = 'general';
    let skills = [], agents = [];
    let selectedAgentId = null;
    let db; // IndexedDB

    // --- IndexedDB for Siphon history ---
    const DB_NAME = 'SiphonDB';
    const STORE_NAME = 'reports';

    function openDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, 1);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    db.createObjectStore(STORE_NAME, { keyPath: 'timestamp' });
                }
            };
            request.onsuccess = (e) => { db = e.target.result; resolve(db); };
            request.onerror = (e) => reject(e);
        });
    }

    function saveSiphonReport(report) {
        if (!db) return;
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        store.put(report);
    }

    async function loadSiphonHistoryFromDB() {
        if (!db) await openDB();
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const request = store.getAll();
        return new Promise((resolve) => {
            request.onsuccess = () => resolve(request.result.sort((a,b) => b.timestamp - a.timestamp));
            request.onerror = () => resolve([]);
        });
    }

    // --- WebSocket ---
    function connect() {
        const proto = location.protocol==='https:'?'wss:':'ws:';
        const port = WS_PORT || location.port || 3721;
        const wsUrl = `${proto}//${location.hostname}:${port}/ws`;
        ws = new WebSocket(wsUrl);
        ws.onopen = ()=>{
            ws.send(JSON.stringify({type:'join', channel: currentChannel}));
            ws.send(JSON.stringify({type:'message', content:'/skills'}));
            ws.send(JSON.stringify({type:'message', content:'/agents'}));
        };
        ws.onmessage = (e)=>{
            try {
                const msg = JSON.parse(e.data);
                switch(msg.type){
                    case 'new_message': appendMessage(msg.message); break;
                    case 'history': renderMessages(msg.messages); break;
                    case 'notification': appendSystemMessage(msg.msg); break;
                    case 'error': appendSystemMessage('X '+msg.msg); break;
                    case 'skills_list': skills=msg.skills; renderSkills(); updateSkillCount(); break;
                    case 'agents_list': agents=msg.agents; renderAgents(); break;
                    case 'ralph_update':
                        updateRalphProgress(msg.data);
                        // Also refresh sessions list
                        ws.send(JSON.stringify({type:'get_ralph_sessions'}));
                        break;
                    case 'ralph_sessions': renderRalphSessions(msg.sessions); break;
                    case 'siphon_history': renderSiphonHistoryFromServer(msg.history); break;
                    case 'global_metrics':
                        document.getElementById('metricCpu').textContent = msg.data.cpu.toFixed(1);
                        document.getElementById('metricMem').textContent = msg.data.memory.toFixed(1);
                        document.getElementById('metricToken').textContent = msg.data.token_rate.toFixed(1);
                        break;
                    case 'agent_loading':
                        const agentEl = document.querySelector(`[data-agent-id="${msg.agent_id}"]`);
                        if (agentEl) {
                            const bar = agentEl.querySelector('.agent-loading-bar');
                            if (bar) {
                                bar.style.width = msg.loading ? '100%' : '0%';
                            }
                        }
                        break;
                    default: console.log('Unhandled WS:', msg);
                }
            } catch(err) {
                console.warn('Failed to parse WebSocket message:', err);
            }
        };
        ws.onclose = ()=>{
            console.log('WebSocket disconnected, reconnecting...');
            setTimeout(connect,2000);
        };
        ws.onerror = (err) => console.error('WebSocket error:', err);
    }

    function sendCommand(cmd) {
        try {
            ws.send(JSON.stringify({type:'message', content:cmd}));
        } catch (e) {
            appendSystemMessage('WebSocket error: ' + e.message);
            return;
        }
        const container = document.getElementById('cliOutput');
        const line = document.createElement('div'); line.className='cli-line';
        const span = document.createElement('span'); span.className='cli-command';
        span.innerHTML = `$ ${cmd}`;
        line.appendChild(span);
        container.appendChild(line);
        container.scrollTop = container.scrollHeight;
    }

    function appendMessage(msg) {
        const container = document.getElementById('cliOutput');
        const line = document.createElement('div'); line.className='cli-line';
        const span = document.createElement('span'); span.className='cli-info';
        const prefix = msg.sender_type==='human' ? 'user' : 'agent';
        span.innerHTML = `[${prefix}] <strong>${msg.sender}</strong>: ${msg.content.replace(/</g,'&lt;').replace(/>/g,'&gt;')}`;
        line.appendChild(span);
        container.appendChild(line);
        container.scrollTop = container.scrollHeight;
        if (msg.sender === 'Siphon' && msg.store_id === 'siphon') {
            saveSiphonReport({ content: msg.content, timestamp: msg.timestamp });
        }
    }

    function appendSystemMessage(text) {
        const container = document.getElementById('cliOutput');
        const line = document.createElement('div'); line.className='cli-line';
        const span = document.createElement('span'); span.className='cli-warning';
        span.innerHTML = '! ' + text;
        line.appendChild(span);
        container.appendChild(line);
        container.scrollTop = container.scrollHeight;
    }

    function renderMessages(messages) {}

    function renderSkills() {
        const list = document.getElementById('combinedSkillList');
        list.innerHTML = '';
        skills.forEach(skill => {
            const div = document.createElement('div'); div.className='skill-item';
            div.innerHTML = `<div class="skill-info"><span class="skill-name">${skill.name}</span><div class="skill-meta">${skill.description}</div></div><div class="skill-actions"><button class="icon-btn view" data-name="${skill.name}">view</button></div>`;
            div.querySelector('.view').onclick = (e) => { e.stopPropagation(); showSkill(skill.name); };
            list.appendChild(div);
        });
        document.getElementById('refTotalCount').innerText = skills.length;
    }

    function updateSkillCount() {
        document.getElementById('skillCountHeader').innerText = skills.length;
        document.getElementById('skillCountFooter').innerText = `${skills.length} skills`;
    }

    function renderAgents() {
        const container = document.getElementById('agentListContainer');
        container.innerHTML = '';
        agents.forEach(a => {
            const div = document.createElement('div');
            div.className = 'skill-item';
            div.style.cursor = 'pointer';
            div.setAttribute('data-agent-id', a.id);
            const status = a.status || 'offline';
            const dotClass = (status === 'online' || status === 'ready') ? 'online' : (status === 'working' ? 'working' : 'offline');
            div.innerHTML = `
                <div class="skill-info">
                    <span class="skill-name"><span class="status-dot ${dotClass}"></span> ${a.name}</span>
                    <div class="skill-meta">${a.model} | ${a.channels ? a.channels.join(', ') : ''}</div>
                    <div class="agent-loading-bar" style="width:0%;height:2px;background:var(--secondary);transition:width 0.3s;border-radius:2px;margin-top:3px;"></div>
                </div>
                <div class="skill-actions">
                    <button class="icon-btn" onclick="event.stopPropagation(); showAgentPopup('${a.id}')">settings</button>
                </div>
            `;
            div.onclick = () => showAgentPopup(a.id);
            container.appendChild(div);
        });
    }

    function renderRalphSessions(sessions) {
        const container = document.getElementById('ralphSessionsContainer');
        container.innerHTML = '';
        if (!sessions || !sessions.length) {
            container.innerHTML = '<div style="color:var(--text-dim);">No active Ralph sessions.</div>';
            return;
        }
        sessions.forEach(s => {
            const div = document.createElement('div');
            div.className = 'skill-item';
            const pct = (s.convergence * 100).toFixed(1);
            div.innerHTML = `
                <div class="skill-info">
                    <span class="skill-name">${s.goal.substring(0, 30)}</span>
                    <div class="skill-meta">Gen ${s.current_gen} | Convergence ${pct}% | Stagnation ${s.stagnation}/3</div>
                </div>
                <div><span class="status-dot ${s.status === 'running' ? 'online' : 'offline'}"></span></div>
            `;
            container.appendChild(div);
        });
    }

    async function renderSiphonHistoryFromServer(history) {
        const container = document.getElementById('siphonHistoryList');
        const dbHistory = await loadSiphonHistoryFromDB();
        const combined = [...history, ...dbHistory];
        const seen = new Set();
        const unique = combined.filter(item => {
            const key = item.timestamp;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
        unique.sort((a,b) => b.timestamp - a.timestamp);
        if (!unique.length) {
            container.innerHTML = '<div style="color:var(--text-dim);">No research reports yet.</div>';
            return;
        }
        container.innerHTML = unique.slice(0, 20).map(item => `
            <div style="border-bottom:1px solid var(--border); padding:8px 0; font-size:10px;">
                <span style="color:var(--text-dim);">${new Date(item.timestamp).toLocaleString()}</span><br>
                <span style="white-space:pre-wrap;">${item.content.substring(0, 200)}</span>
            </div>
        `).join('');
    }

    async function loadSiphonHistory() {
        ws.send(JSON.stringify({type:'get_siphon_history'}));
    }

    function showSkill(name) {
        const skill = skills.find(s => s.name === name);
        if (!skill) return;
        document.getElementById('viewModalTitle').innerText = `${name}.skill.md`;
        document.getElementById('viewModalContent').innerText = JSON.stringify(skill, null, 2);
        document.getElementById('viewModal').classList.add('active');
    }

    function populateQuickActions() {
        const container = document.getElementById('combinedQuickActions');
        const actions = ['/skills', '/agents', '/ralph', '/abstract', '/stack list', '/reverse_skill', '/status', '/help'];
        container.innerHTML = actions.map(cmd => `<div class="quick-cmd" data-cmd="${cmd}">${cmd}</div>`).join('');
        container.querySelectorAll('.quick-cmd').forEach(el => {
            el.onclick = () => sendCommand(el.dataset.cmd);
        });
    }

    function updateRalphProgress(data) {
        const container = document.getElementById('ralphProgressDisplay');
        if (!data || typeof data !== 'object') {
            container.innerHTML = 'No active Ralph session';
            return;
        }
        const statusIcon = data.status === 'running' ? 'running' : 'done';
        const pct = (data.convergence * 100).toFixed(1);
        container.innerHTML = `
            ${statusIcon} Generation ${data.current_gen} / ${MAX_GENERATIONS}
            | Convergence: ${pct}%
            | Stagnation: ${data.stagnation}/3
            <br><span style="color:var(--text-dim);">Best: ${data.best_spec.substring(0, 60)}...</span>
        `;
    }

    // ENHANCED: showAgentPopup with loading, retry, and refresh
    // FIX: initial fetch and retry both force server refresh
    function showAgentPopup(agentId) {
        selectedAgentId = agentId;
        const agent = agents.find(a => a.id === agentId);
        if (!agent) return;
        document.getElementById('popupName').value = agent.name || '';
        document.getElementById('popupSystemPrompt').value = agent.system_prompt || '';
        document.getElementById('popupChannels').value = (agent.channels || []).join(',');

        const modelSelect = document.getElementById('popupModel');
        modelSelect.innerHTML = '';
        const manualContainer = document.getElementById('manualModelContainer');
        manualContainer.style.display = 'none';
        const loading = document.getElementById('modelLoading');
        const retryBtn = document.getElementById('retryModelsBtn');
        const refreshBtn = document.getElementById('refreshModelsBtn');
        loading.style.display = 'block';
        retryBtn.style.display = 'none';
        refreshBtn.style.display = 'inline-block';

        // Function to populate dropdown or show manual
        function populateModels(models) {
            loading.style.display = 'none';
            if (models && models.length > 0) {
                models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.name;
                    if (m.name === agent.model) opt.selected = true;
                    modelSelect.appendChild(opt);
                });
                // Add default model as first option if not present
                const defaultModel = 'qwen2.5:0.5b';
                if (!models.some(m => m.name === defaultModel)) {
                    const opt = document.createElement('option');
                    opt.value = defaultModel;
                    opt.textContent = defaultModel + ' (default)';
                    if (agent.model === defaultModel) opt.selected = true;
                    modelSelect.prepend(opt);
                }
            } else {
                // No models: show manual input and placeholder
                manualContainer.style.display = 'block';
                modelSelect.innerHTML = '<option value="">No models found – type manually</option>';
                // Pre-fill manual input with agent's current model if not default
                const currentModel = agent.model || '';
                if (currentModel && currentModel !== 'No models found – type manually') {
                    document.getElementById('manualModelInput').value = currentModel;
                }
            }
        }

        // Fetch models with cache-busting and force server refresh
        function fetchModels(forceRefresh = false) {
            loading.style.display = 'block';
            retryBtn.style.display = 'none';
            const url = forceRefresh ? '/api/tags?refresh=true&_=' + Date.now() : '/api/tags?_=' + Date.now();
            fetch(url)
                .then(r => r.json())
                .then(data => {
                    if (data.models) {
                        populateModels(data.models);
                    } else {
                        populateModels([]);
                    }
                })
                .catch(err => {
                    loading.style.display = 'none';
                    retryBtn.style.display = 'inline-block';
                    // Show manual input anyway
                    manualContainer.style.display = 'block';
                    modelSelect.innerHTML = '<option value="">Error loading models – type manually</option>';
                    console.warn('Model fetch error:', err);
                });
        }

        // Initial fetch - force server refresh to avoid stale cache
        fetchModels(true);
        // Retry handler - also force server refresh
        retryBtn.onclick = function() { fetchModels(true); };
        // Refresh button forces server refresh as well
        refreshBtn.onclick = function() { fetchModels(true); };

        document.getElementById('agentPopup').style.display = 'block';
    }

    function closeAgentPopup() {
        document.getElementById('agentPopup').style.display = 'none';
        selectedAgentId = null;
    }

    function useManualModel() {
        const input = document.getElementById('manualModelInput');
        const model = input.value.trim();
        if (model) {
            const select = document.getElementById('popupModel');
            const opt = document.createElement('option');
            opt.value = model;
            opt.textContent = model;
            opt.selected = true;
            select.appendChild(opt);
            document.getElementById('manualModelContainer').style.display = 'none';
        }
    }

    // FIX: saveAgentSettings – auto‑use manual input if dropdown invalid, fallback to default
    function saveAgentSettings() {
        if (!selectedAgentId) return;
        const name = document.getElementById('popupName').value.trim();
        let model = document.getElementById('popupModel').value;
        const manualContainer = document.getElementById('manualModelContainer');
        const manualInput = document.getElementById('manualModelInput');

        // If manual container is visible, use manual input
        if (manualContainer.style.display !== 'none') {
            const manualModel = manualInput.value.trim();
            if (manualModel) {
                model = manualModel;
                // Optionally add it to dropdown for future saves
                const opt = document.createElement('option');
                opt.value = manualModel;
                opt.textContent = manualModel;
                opt.selected = true;
                document.getElementById('popupModel').appendChild(opt);
                manualContainer.style.display = 'none';
            } else {
                alert('Please enter a model name manually or select one from the list.');
                return;
            }
        }

        // If model is still empty or placeholder, fallback to default
        if (!model || model === 'No models found – type manually') {
            model = 'qwen2.5:0.5b';
        }

        const systemPrompt = document.getElementById('popupSystemPrompt').value.trim();
        const channels = document.getElementById('popupChannels').value.split(',').map(c => c.trim()).filter(c => c);
        if (!name) { alert('Name is required'); return; }

        sendCommand(`/update_agent ${selectedAgentId} name "${name}"`);
        sendCommand(`/update_agent ${selectedAgentId} model ${model}`);
        sendCommand(`/update_agent ${selectedAgentId} system_prompt "${systemPrompt}"`);
        closeAgentPopup();
        appendSystemMessage(`Agent ${name} updated successfully`);
    }

    function deleteSelectedAgent() {
        if (!selectedAgentId) return;
        if (confirm(`Delete agent ${selectedAgentId}? This cannot be undone.`)) {
            sendCommand(`/remove_agent ${selectedAgentId}`);
            closeAgentPopup();
        }
    }

    function updateRalphSessions() {
        ws.send(JSON.stringify({type:'get_ralph_sessions'}));
    }

    // --- Event listeners ---
    document.getElementById('cliInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const val = e.target.value.trim();
            if (val) sendCommand(val);
            e.target.value = '';
        }
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            document.getElementById(`${tabId}Tab`).classList.add('active');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (tabId === 'ralph') updateRalphSessions();
            if (tabId === 'siphon') loadSiphonHistory();
        });
    });

    document.getElementById('closeViewBtn').addEventListener('click', () => document.getElementById('viewModal').classList.remove('active'));
    document.getElementById('reloadSkillsBtn').addEventListener('click', () => sendCommand('/reload_skills'));
    document.getElementById('createSkillBtn').addEventListener('click', () => {
        const name = prompt('Enter skill name:');
        if (name) sendCommand(`/create_skill ${name}`);
    });

    document.getElementById('spawnAgentBtn').addEventListener('click', () => {
        const name = prompt('Agent name:', 'New Agent');
        if (!name) return;
        const model = prompt('Model:', 'qwen2.5:0.5b') || 'qwen2.5:0.5b';
        const systemPrompt = prompt('System prompt:', 'You are a helpful assistant.') || 'You are a helpful assistant.';
        const channelsInput = prompt('Channels (comma-separated, e.g., general,code):', 'general');
        const channels = channelsInput ? channelsInput.split(',').map(c => c.trim()) : ['general'];
        ws.send(JSON.stringify({
            type: 'spawn_agent',
            name, model, system_prompt: systemPrompt,
            channels
        }));
    });

    document.getElementById('newSiphonBtn').addEventListener('click', () => {
        const query = prompt('Enter research topic or URL:');
        if (query) sendCommand(`/siphon ${query}`);
    });

    document.getElementById('newRalphBtn').addEventListener('click', () => {
        const topic = prompt('Enter topic for Ralph evolution:');
        if (topic) sendCommand(`/ralph ${topic}`);
    });

    document.getElementById('cliOutput').addEventListener('click', (e) => {
        const target = e.target.closest('.cli-link');
        if (target) {
            const cmd = target.dataset.cmd;
            if (cmd) sendCommand(cmd);
        }
    });

    populateQuickActions();
    openDB().then(() => console.log('IndexedDB ready'));
    connect();
</script>
</body>
</html>
'''

# ==============================================================================
# SECTION 19: MAIN ENTRY POINT
# PURPOSE: Parses command‑line arguments and launches either the web server
#          or the CLI mode. Handles shutdown signals gracefully.
# ==============================================================================
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode (no web UI)')
    parser.add_argument('--port', type=int, help='Override server port')
    args = parser.parse_args()

    if args.port:
        settings.port = args.port

    if args.cli:
        cli = CLIMode()
        await cli.run()
    else:
        server = DashboardServer()
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler():
            logger.info("Shutdown signal received")
            server.shutdown()
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                pass

        server_task = asyncio.create_task(server.start())
        try:
            await shutdown_event.wait()
            logger.info("Shutting down gracefully...")
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            logger.info("Shutdown complete")
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Exited.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
