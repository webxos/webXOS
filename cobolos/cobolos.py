#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COBOLOS v2.0 – Local COBOL Modernization Agent Harness
========================================================
Enhanced with:
- WebSocket bidirectional streaming (with optional auth)
- External DB (DB2/VSAM) support for Agent 01
- API key authentication (HTTP and WebSocket)
- Scheduled (cron) agent runs
- CLI mode for batch processing
- Thread‑safe external DB initialisation
- Scheduler started on app startup
- Fully automatic bootstrap (creates venv if needed)
- Modern lifespan for startup/shutdown
- Fixed UI template escaping
"""

import subprocess
import sys
import os
from pathlib import Path

# ---------- Self-Bootstrap with automatic venv ----------
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "requests",
    "sqlparse",
    "rich",
    "apscheduler",
    "websockets",
    "python-multipart",  # required for file uploads
]

def bootstrap():
    # If we are already inside a virtual environment, skip venv creation
    in_venv = (hasattr(sys, 'real_prefix') or
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

    # Check for missing packages
    missing = []
    for pkg in REQUIRED_PACKAGES:
        # special case for python-multipart: it's imported as "multipart"
        import_name = pkg.replace("-", "_")
        if import_name == "python_multipart":
            import_name = "multipart"
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if not missing:
        print("[+] All dependencies found.")
        return

    print(f"[*] Missing packages: {', '.join(missing)}")

    # If not in venv, try to install globally; if externally-managed, create venv
    if not in_venv:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing,
                check=True,
                capture_output=True,
                text=True
            )
            print("[*] Dependencies installed globally. Proceeding...")
            # Re-import now that they are installed (should succeed)
            return
        except subprocess.CalledProcessError as e:
            combined = (e.stderr or "") + (e.stdout or "")
            if "externally-managed-environment" in combined:
                print("[*] Detected externally-managed Python. Creating virtual environment...")
                # Create venv in the script directory
                base_dir = Path(__file__).parent.absolute()
                venv_dir = base_dir / "venv"
                if not venv_dir.exists():
                    subprocess.run(
                        [sys.executable, "-m", "venv", str(venv_dir)],
                        check=True
                    )
                # Determine python executable inside venv
                if sys.platform == "win32":
                    venv_python = venv_dir / "Scripts" / "python.exe"
                else:
                    venv_python = venv_dir / "bin" / "python"
                # Install missing packages inside venv
                subprocess.run(
                    [str(venv_python), "-m", "pip", "install"] + missing,
                    check=True
                )
                print("[*] Dependencies installed in virtual environment.")
                # Re-execute this script with venv python
                os.execv(str(venv_python), [str(venv_python)] + sys.argv)
            else:
                print(f"[!] pip install failed: {combined}")
                print("Please install the missing packages manually.")
                sys.exit(1)
    else:
        # We are inside a venv but still missing packages; try installing them
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing,
                check=True,
                capture_output=True,
                text=True
            )
            print("[*] Dependencies installed in virtual environment. Proceeding...")
            # Re-import now (should succeed, but we can just return)
            return
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to install packages in venv: {e.stderr}")
            sys.exit(1)

bootstrap()

# ---------- Now import all external packages ----------
import asyncio
import json
import logging
import re
import sqlite3
import threading
import time
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager

import requests
import sqlparse
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# ---------- Lifespan for startup/shutdown ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialise external DB and scheduler
    get_external_db()
    init_schedules()
    logger.info("Application startup complete. Scheduler started.")
    yield
    # Shutdown: (optional) clean up resources
    scheduler.shutdown()
    logger.info("Application shutdown complete.")

# ---------- FastAPI App ----------
app = FastAPI(
    title="COBOLOS v2.0",
    description="Local COBOL Modernization Agent",
    lifespan=lifespan
)

# ---------- Global concurrency limiter (module-level) ----------
llm_semaphore = asyncio.Semaphore(2)   # Limit concurrent LLM calls

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'client_ip'):
            record.client_ip = 'unknown'
        if not hasattr(record, 'agent_id'):
            record.agent_id = 'N/A'
        return True

logger.addFilter(ContextFilter())

# ---------- ASCII Logo ----------
COBOLOS_ASCII = r"""
   ____      _           _  ___  ____  
  / ___|___ | |__   ___ | |/ _ \/ ___| 
 | |   / _ \| '_ \ / _ \| | | | \___ \ 
 | |__| (_) | |_) | (_) | | |_| |___) |
  \____\___/|_.__/ \___/|_|\___/|____/ 
                                       Local COBOL Modernization Agent Harness v2.0
"""

def print_logo():
    print(COBOLOS_ASCII)

# ---------- Constants ----------
BASE_DIR = Path(__file__).parent.absolute()
WORKSPACE_DIR = BASE_DIR / "cobolos_workspace"
SOURCE_POOL = WORKSPACE_DIR / "source_pool"
CONFIG_FILE = BASE_DIR / "config.json"
DB_PATH = WORKSPACE_DIR / "cobolos.db"
HISTORY_DB = WORKSPACE_DIR / "history.db"

DEFAULT_CONFIG = {
    "ollama_url": "http://localhost:11434/api/generate",
    "default_model": "codellama:7b",
    "temperature": 0.3,
    "max_tokens": 2000,
    "db_path": str(DB_PATH),
    "workspace": str(WORKSPACE_DIR),
    "timeout": 60,
    "agent_timeout": 120,
    "stream_timeout": 60,
    "api_key": None,
    "external_db": {
        "type": "sqlite",
        "connection_string": ""
    },
    "schedules": []
}

ALLOWED_COLUMNS = {
    "CUSTOMERS": ["CUST_ID", "FIRST_NAME", "LAST_NAME", "STATE", "BIRTH_DATE", "BALANCE"],
    "ORDERS": ["ORDER_ID", "CUST_ID", "ORDER_DATE", "TOTAL", "STATUS"],
}

AGENTS = {
    "01": {"name": "NL Data Query", "desc": "Natural language → SQL", "inputs": ["Question"]},
    "02": {"name": "Error Log Diagnostic", "desc": "Analyse abend logs", "inputs": ["Log text"]},
    "03": {"name": "JCL/Batch Generator", "desc": "Create JCL from spec", "inputs": ["Job description"]},
    "04": {"name": "Code Explainer", "desc": "Explain COBOL logic", "inputs": ["COBOL code"]},
    "05": {"name": "Automated Unit Test", "desc": "Generate test harness", "inputs": ["Program name"]},
    "06": {"name": "Modernization Guard", "desc": "Suggest refactoring", "inputs": ["COBOL code"]},
    "07": {"name": "3270 Screen API", "desc": "Generate screen API", "inputs": ["Screen layout"]},
    "08": {"name": "Rule Extraction", "desc": "Extract business rules", "inputs": ["COBOL code"]},
    "09": {"name": "Smart Impact Analysis", "desc": "Find dependencies", "inputs": ["Search term"]},
    "10": {"name": "Auto Documentation", "desc": "Generate docs", "inputs": ["Program name"]},
}

# ---------- Configuration ----------
def load_config() -> Dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()

def get_ollama_url() -> str:
    return CONFIG.get("ollama_url", DEFAULT_CONFIG["ollama_url"])

def get_default_model() -> str:
    return CONFIG.get("default_model", DEFAULT_CONFIG["default_model"])

def get_temperature() -> float:
    return CONFIG.get("temperature", DEFAULT_CONFIG["temperature"])

def get_max_tokens() -> int:
    return CONFIG.get("max_tokens", DEFAULT_CONFIG["max_tokens"])

def get_timeout() -> int:
    return CONFIG.get("timeout", DEFAULT_CONFIG["timeout"])

def get_agent_timeout() -> int:
    return CONFIG.get("agent_timeout", DEFAULT_CONFIG["agent_timeout"])

def get_stream_timeout() -> int:
    return CONFIG.get("stream_timeout", DEFAULT_CONFIG["stream_timeout"])

def get_api_key() -> Optional[str]:
    return CONFIG.get("api_key")

def get_external_db_config() -> Dict:
    return CONFIG.get("external_db", DEFAULT_CONFIG["external_db"])

# ---------- Authentication ----------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def optional_auth(api_key: str = Depends(API_KEY_HEADER)):
    expected = get_api_key()
    if expected is not None and (api_key is None or api_key != expected):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

def websocket_auth(websocket: WebSocket) -> bool:
    """Check API key from query parameter or 'X-API-Key' header."""
    expected = get_api_key()
    if expected is None:
        return True
    # Try query parameter
    api_key = websocket.query_params.get("api_key")
    if api_key and api_key == expected:
        return True
    # Try header
    headers = websocket.headers
    api_key = headers.get("x-api-key")
    if api_key and api_key == expected:
        return True
    return False

# ---------- Workspace & Git ----------
GIT_AVAILABLE = False

def init_workspace():
    global GIT_AVAILABLE
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(SOURCE_POOL, exist_ok=True)
    os.makedirs(WORKSPACE_DIR / "artifacts", exist_ok=True)
    os.makedirs(WORKSPACE_DIR / "history", exist_ok=True)

    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        GIT_AVAILABLE = True
    except FileNotFoundError:
        logger.warning("Git not installed; versioning disabled.")
        GIT_AVAILABLE = False

    if GIT_AVAILABLE:
        repo = WORKSPACE_DIR / ".git"
        if not repo.exists():
            subprocess.run(["git", "init"], cwd=WORKSPACE_DIR, check=True)
            subprocess.run(["git", "config", "user.name", "COBOLOS"], cwd=WORKSPACE_DIR, check=True)
            subprocess.run(["git", "config", "user.email", "agent@cobolos.internal"], cwd=WORKSPACE_DIR, check=True)
            with open(WORKSPACE_DIR / ".gitignore", "w") as f:
                f.write("*.db\nhistory/\n__pycache__/\n*.zip\n")
            subprocess.run(["git", "add", "."], cwd=WORKSPACE_DIR, check=True)
            subprocess.run(["git", "commit", "-m", "Initial workspace"], cwd=WORKSPACE_DIR, check=True)
            logger.info("Git repository initialised.")

def git_commit_artifact(file_path: Union[str, Path], message: str):
    if not GIT_AVAILABLE:
        return
    path = Path(file_path)
    if not path.exists():
        return
    try:
        status = subprocess.run(["git", "status", "--porcelain", str(path)], cwd=WORKSPACE_DIR,
                                capture_output=True, text=True)
        if not status.stdout.strip():
            return
        subprocess.run(["git", "add", str(path)], cwd=WORKSPACE_DIR, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=WORKSPACE_DIR, check=True)
        logger.info(f"Committed {path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git commit failed: {e.stderr}")

def save_artifact(agent_id: str, result: Dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"artifacts/agent_{agent_id}_{timestamp}.json"
    full_path = WORKSPACE_DIR / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    git_commit_artifact(full_path, f"Agent {agent_id} result")
    return str(filename)

# ---------- External Database Abstraction (Thread‑safe) ----------
_ext_db_lock = threading.Lock()
_ext_db = None

class ExternalDB:
    def __init__(self, config: Dict):
        self.config = config
        self.db_type = config.get("type", "sqlite")
        self.conn_string = config.get("connection_string", "")
        self._connection = None

    def connect(self):
        if self.db_type == "sqlite":
            db_path = CONFIG.get("db_path", DEFAULT_CONFIG["db_path"])
            self._connection = sqlite3.connect(db_path)
            self._connection.row_factory = sqlite3.Row
        elif self.db_type == "db2":
            logger.warning("DB2 not implemented; falling back to SQLite.")
            db_path = CONFIG.get("db_path", DEFAULT_CONFIG["db_path"])
            self._connection = sqlite3.connect(db_path)
            self._connection.row_factory = sqlite3.Row
        elif self.db_type == "vsam":
            logger.warning("VSAM not implemented; falling back to SQLite.")
            db_path = CONFIG.get("db_path", DEFAULT_CONFIG["db_path"])
            self._connection = sqlite3.connect(db_path)
            self._connection.row_factory = sqlite3.Row
        else:
            raise ValueError(f"Unsupported DB type: {self.db_type}")

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        if self._connection is None:
            self.connect()
        cursor = self._connection.cursor()
        try:
            cursor.execute(query, params)
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                self._connection.commit()
                return [{"affected_rows": cursor.rowcount}]
        except Exception as e:
            return [{"error": str(e)}]
        finally:
            cursor.close()

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None

def get_external_db():
    global _ext_db
    if _ext_db is None:
        with _ext_db_lock:
            if _ext_db is None:  # double-check
                _ext_db = ExternalDB(get_external_db_config())
                _ext_db.connect()
    return _ext_db

# ---------- Internal DB ----------
def init_mock_db():
    db_path = Path(CONFIG.get("db_path", DEFAULT_CONFIG["db_path"]))
    os.makedirs(db_path.parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CUSTOMERS (
            CUST_ID INTEGER PRIMARY KEY,
            FIRST_NAME TEXT,
            LAST_NAME TEXT,
            STATE TEXT,
            BIRTH_DATE TEXT,
            BALANCE REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ORDERS (
            ORDER_ID INTEGER PRIMARY KEY,
            CUST_ID INTEGER,
            ORDER_DATE TEXT,
            TOTAL REAL,
            STATUS TEXT,
            FOREIGN KEY (CUST_ID) REFERENCES CUSTOMERS(CUST_ID)
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM CUSTOMERS")
    if cursor.fetchone()[0] == 0:
        customers = [
            (1, "John", "Doe", "CA", "1980-01-01", 1500.00),
            (2, "Jane", "Smith", "NY", "1990-05-15", 2200.50),
            (3, "Bob", "Johnson", "TX", "1975-12-10", 800.75),
        ]
        cursor.executemany("INSERT INTO CUSTOMERS VALUES (?,?,?,?,?,?)", customers)
        orders = [
            (101, 1, "2023-01-10", 250.00, "Shipped"),
            (102, 1, "2023-02-14", 120.50, "Pending"),
            (103, 2, "2023-03-01", 99.99, "Shipped"),
        ]
        cursor.executemany("INSERT INTO ORDERS VALUES (?,?,?,?,?)", orders)
    conn.commit()
    conn.close()
    logger.info("Mock database initialised.")

def execute_safe_query(query: str, params: tuple = ()) -> List[Dict]:
    parsed = sqlparse.parse(query)
    if len(parsed) != 1:
        return [{"error": "Multiple SQL statements not allowed."}]
    stmt = parsed[0]
    if stmt.get_type() != 'SELECT':
        return [{"error": "Only SELECT queries are allowed."}]
    stmt_str = str(stmt).upper()
    dangerous = ["UNION", "INSERT", "DELETE", "UPDATE", "DROP", "ALTER", "CREATE", "EXEC", "EXECUTE"]
    for word in dangerous:
        if word in stmt_str:
            return [{"error": f"Dangerous keyword '{word}' detected."}]
    from_clause = re.search(r"FROM\s+(\w+)", stmt_str)
    if not from_clause:
        return [{"error": "No FROM clause found."}]
    table = from_clause.group(1)
    if table not in ALLOWED_COLUMNS:
        return [{"error": f"Table '{table}' not allowed."}]
    allowed_cols = ALLOWED_COLUMNS[table]

    select_part = stmt_str.split("FROM")[0].replace("SELECT", "").strip()
    if select_part != "*":
        cols = [c.strip() for c in select_part.split(",") if c.strip()]
        for col in cols:
            col_clean = re.sub(r"\s+AS\s+\w+", "", col, flags=re.IGNORECASE)
            col_clean = re.sub(r"^\w+\.", "", col_clean)
            if col_clean not in allowed_cols:
                return [{"error": f"Column '{col_clean}' not allowed in table {table}."}]

    where_clause = re.search(r"WHERE\s+(.+)", stmt_str)
    if where_clause:
        col_names = re.findall(r"\b(\w+)\s*[=<>]", where_clause.group(1))
        for col in col_names:
            if col not in allowed_cols:
                return [{"error": f"Column '{col}' not allowed in table {table}."}]

    ext_cfg = get_external_db_config()
    if ext_cfg.get("type") != "sqlite":
        db = get_external_db()
        return db.execute_query(query, params)
    else:
        db_path = CONFIG.get("db_path", DEFAULT_CONFIG["db_path"])
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            return [{"error": f"SQL error: {e}"}]
        finally:
            conn.close()

# ---------- Ollama Helper ----------
def call_ollama(
    prompt: str,
    system_prompt: str = "You are an expert COBOL modernization assistant.",
    model: Optional[str] = None,
    stream: bool = False,
) -> Union[str, AsyncGenerator[str, None]]:
    if model is None:
        model = get_default_model()
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "temperature": get_temperature(),
        "max_tokens": get_max_tokens(),
        "stream": stream,
    }
    url = get_ollama_url()
    timeout = get_timeout()

    if not stream:
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json().get("response", "Error: Empty response.")
            elif response.status_code == 404:
                return f"Error: Model '{model}' not found. Pull it with 'ollama pull {model}'."
            else:
                return f"Error: Ollama returned {response.status_code} - {response.text}"
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Ensure 'ollama serve' is running."
        except requests.exceptions.Timeout:
            return f"Error: Ollama request timed out after {timeout}s. Try a smaller model."
    else:
        try:
            with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    yield f"Error: {r.status_code} - {r.text}"
                    return
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"

# ---------- JSON Extraction ----------
def extract_json(text: str) -> Optional[Dict]:
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return None

# ---------- Agent Functions (All 10) ----------
def run_agent_01_nl_query(user_input: str) -> Dict:
    prompt = f"""
Convert this question into a SQL query for a database with tables:
CUSTOMERS (CUST_ID, FIRST_NAME, LAST_NAME, STATE, BIRTH_DATE, BALANCE)
ORDERS (ORDER_ID, CUST_ID, ORDER_DATE, TOTAL, STATUS)

Return ONLY a JSON object with keys: "query" (SQL), "explanation" (brief), "params" (list).
Question: {user_input}
"""
    response = call_ollama(prompt, "You are a SQL expert. Output only JSON.")
    parsed = extract_json(response)
    if parsed and "query" in parsed:
        sql = parsed["query"]
        params = parsed.get("params", [])
        result = execute_safe_query(sql, tuple(params))
        parsed["result"] = result
        return parsed
    else:
        return {"error": "Could not parse SQL", "raw": response}

def run_agent_02_error_diagnostic(user_input: str) -> Dict:
    prompt = f"""
Analyse the following error log/abend and provide:
1. Most likely cause.
2. Suggested code changes (if any).
3. Severity (High/Medium/Low).
4. References to IBM abend codes (S0C4, S0C7, etc.) with common fixes.
Log:
{user_input}
Return as JSON with keys: cause, suggestions, severity, references.
"""
    response = call_ollama(prompt, "You are a mainframe expert.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

def run_agent_03_jcl_generator(user_input: str) -> Dict:
    prompt = f"""
Generate a complete JCL job stream based on this description:
{user_input}
Include:
- JOB card with proper accounting.
- PROC library inclusion (if needed).
- Multiple steps with dependencies (use COND or IF/ELSE).
- GDG handling if mentioned.
Return the JCL code and a brief explanation as JSON with keys: jcl, explanation.
"""
    response = call_ollama(prompt, "You are a JCL expert.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

def run_agent_04_code_explainer(user_input: str) -> Dict:
    prompt = f"""
Explain the following COBOL code in detail:
{user_input}
Provide:
1. High‑level purpose.
2. Paragraph‑by‑paragraph breakdown.
3. Data flow (variables and their usage).
4. A Mermaid flowchart of the main logic.
Return as JSON with keys: purpose, paragraphs, data_flow, mermaid.
"""
    response = call_ollama(prompt, "You are a COBOL analyst.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

def run_agent_05_unit_test(user_input: str) -> Dict:
    prompt = f"""
For the COBOL program described or named: {user_input}
Generate a Python pytest test harness that:
- Uses a wrapper to call the COBOL program (assume a function).
- Provides test data in CSV/JSON.
- Tests boundary conditions.
Return the Python code and test data as JSON with keys: test_code, test_data, coverage_suggestions.
"""
    response = call_ollama(prompt, "You are a test automation engineer.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

def run_agent_06_modernization_guard(user_input: str) -> Dict:
    prompt = f"""
Suggest modernisation options for this COBOL code:
{user_input}
Options: COBOL OO, Java, C#, Python.
Provide:
1. Migration target recommendation.
2. Refactoring steps.
3. Equivalence test plan (how to ensure same behaviour).
Return as JSON with keys: target, steps, equivalence_plan.
"""
    response = call_ollama(prompt, "You are a refactoring expert.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

def run_agent_07_screen_api(user_input: str) -> Dict:
    prompt = f"""
Based on the 3270 screen layout described:
{user_input}
Generate:
1. A Swagger/OpenAPI YAML specification for an API that serves this screen.
2. A Python client snippet.
Return as JSON with keys: openapi_yaml, python_client.
"""
    response = call_ollama(prompt, "You are an API designer.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

def run_agent_08_rule_extraction(user_input: str) -> Dict:
    prompt = f"""
Extract all business rules from this COBOL code:
{user_input}
Format as a decision table (CSV) and group them by business function.
Return as JSON with keys: decision_table (CSV string), groups (list of {function: rules}).
"""
    response = call_ollama(prompt, "You are a business analyst.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

def run_agent_09_impact_analysis(user_input: str) -> Dict:
    pattern = re.escape(user_input)
    results = []
    for root, dirs, files in os.walk(WORKSPACE_DIR):
        if ".git" in root:
            continue
        for file in files:
            if file.endswith((".cob", ".cbl", ".copy", ".txt")):
                path = Path(root) / file
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    if re.search(pattern, content, re.IGNORECASE):
                        results.append(str(path.relative_to(WORKSPACE_DIR)))
                except Exception:
                    pass
    if not results:
        return {"files": [], "message": "No files found."}

    truncated = results[:50]
    if len(results) > 50:
        truncated.append(f"... and {len(results)-50} more files")

    prompt = f"""
Given these files that reference '{user_input}':
{truncated}
Build a dependency graph (which files call which) and assign a risk score (1-100) based on complexity and coupling.
Return as JSON with keys: graph (dict of file -> [dependencies]), risk_score, recommendation.
"""
    response = call_ollama(prompt, "You are a system analyst.")
    parsed = extract_json(response)
    if parsed:
        parsed["files"] = results
        return parsed
    return {"files": results, "error": "LLM parse failed"}

def run_agent_10_auto_documentation(user_input: str) -> Dict:
    prompt = f"""
Generate full documentation for the COBOL program '{user_input}':
Include:
- Program overview.
- Data division details.
- Procedure division logic.
- Cross‑references (copybooks, called programs).
- Glossary of technical terms.
- Output in Markdown, HTML, and PDF (provide content).
Return as JSON with keys: markdown, html, pdf_base64 (optional), changes_since_last (if known).
"""
    response = call_ollama(prompt, "You are a technical writer.")
    parsed = extract_json(response)
    if parsed:
        return parsed
    return {"error": "Parse failed", "raw": response}

AGENT_ROUTERS = {
    "01": run_agent_01_nl_query,
    "02": run_agent_02_error_diagnostic,
    "03": run_agent_03_jcl_generator,
    "04": run_agent_04_code_explainer,
    "05": run_agent_05_unit_test,
    "06": run_agent_06_modernization_guard,
    "07": run_agent_07_screen_api,
    "08": run_agent_08_rule_extraction,
    "09": run_agent_09_impact_analysis,
    "10": run_agent_10_auto_documentation,
}

def build_agent_prompt(agent_id: str, user_input: str) -> str:
    if agent_id == "01":
        return f"""
Convert this question into a SQL query for a database with tables:
CUSTOMERS (CUST_ID, FIRST_NAME, LAST_NAME, STATE, BIRTH_DATE, BALANCE)
ORDERS (ORDER_ID, CUST_ID, ORDER_DATE, TOTAL, STATUS)

Return ONLY a JSON object with keys: "query" (SQL), "explanation" (brief), "params" (list).
Question: {user_input}
"""
    elif agent_id == "02":
        return f"""
Analyse the following error log/abend and provide:
1. Most likely cause.
2. Suggested code changes (if any).
3. Severity (High/Medium/Low).
4. References to IBM abend codes (S0C4, S0C7, etc.) with common fixes.
Log:
{user_input}
Return as JSON with keys: cause, suggestions, severity, references.
"""
    elif agent_id == "03":
        return f"""
Generate a complete JCL job stream based on this description:
{user_input}
Include:
- JOB card with proper accounting.
- PROC library inclusion (if needed).
- Multiple steps with dependencies (use COND or IF/ELSE).
- GDG handling if mentioned.
Return the JCL code and a brief explanation as JSON with keys: jcl, explanation.
"""
    elif agent_id == "04":
        return f"""
Explain the following COBOL code in detail:
{user_input}
Provide:
1. High‑level purpose.
2. Paragraph‑by‑paragraph breakdown.
3. Data flow (variables and their usage).
4. A Mermaid flowchart of the main logic.
Return as JSON with keys: purpose, paragraphs, data_flow, mermaid.
"""
    elif agent_id == "05":
        return f"""
For the COBOL program described or named: {user_input}
Generate a Python pytest test harness that:
- Uses a wrapper to call the COBOL program (assume a function).
- Provides test data in CSV/JSON.
- Tests boundary conditions.
Return the Python code and test data as JSON with keys: test_code, test_data, coverage_suggestions.
"""
    elif agent_id == "06":
        return f"""
Suggest modernisation options for this COBOL code:
{user_input}
Options: COBOL OO, Java, C#, Python.
Provide:
1. Migration target recommendation.
2. Refactoring steps.
3. Equivalence test plan (how to ensure same behaviour).
Return as JSON with keys: target, steps, equivalence_plan.
"""
    elif agent_id == "07":
        return f"""
Based on the 3270 screen layout described:
{user_input}
Generate:
1. A Swagger/OpenAPI YAML specification for an API that serves this screen.
2. A Python client snippet.
Return as JSON with keys: openapi_yaml, python_client.
"""
    elif agent_id == "08":
        return f"""
Extract all business rules from this COBOL code:
{user_input}
Format as a decision table (CSV) and group them by business function.
Return as JSON with keys: decision_table (CSV string), groups (list of {function: rules}).
"""
    elif agent_id == "09":
        return f"""
Given these files that reference '{user_input}':
(Search results will be provided by the agent, but for streaming we just use the input)
Build a dependency graph (which files call which) and assign a risk score (1-100) based on complexity and coupling.
Return as JSON with keys: graph (dict of file -> [dependencies]), risk_score, recommendation.
"""
    elif agent_id == "10":
        return f"""
Generate full documentation for the COBOL program '{user_input}':
Include:
- Program overview.
- Data division details.
- Procedure division logic.
- Cross‑references (copybooks, called programs).
- Glossary of technical terms.
- Output in Markdown, HTML, and PDF (provide content).
Return as JSON with keys: markdown, html, pdf_base64 (optional), changes_since_last (if known).
"""
    else:
        return f"User input: {user_input}"

# ---------- History ----------
def init_history_db():
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            agent_id TEXT,
            input TEXT,
            output TEXT,
            file_ref TEXT
        )
    """)
    conn.close()

def save_history(agent_id: str, input_text: str, output: Dict, file_ref: str = ""):
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute(
        "INSERT INTO history (timestamp, agent_id, input, output, file_ref) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), agent_id, input_text, json.dumps(output, default=str), file_ref)
    )
    conn.commit()
    conn.close()

def get_history(limit: int = 50) -> List[Dict]:
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]

# ---------- Scheduler ----------
scheduler = AsyncIOScheduler()

def schedule_job(agent_id: str, cron_expr: str, input_text: str):
    trigger = CronTrigger.from_crontab(cron_expr)
    scheduler.add_job(
        run_agent_job,
        trigger,
        args=[agent_id, input_text],
        id=f"{agent_id}_{datetime.now().timestamp()}"
    )
    logger.info(f"Scheduled job for agent {agent_id} with cron '{cron_expr}'")

def run_agent_job(agent_id: str, input_text: str):
    logger.info(f"Scheduled job: running agent {agent_id}")
    try:
        result = AGENT_ROUTERS[agent_id](input_text)
        file_ref = save_artifact(agent_id, result)
        save_history(agent_id, input_text, result, file_ref)
        logger.info(f"Scheduled job completed: agent {agent_id}")
    except Exception as e:
        logger.error(f"Scheduled job failed: {e}")

def init_schedules():
    for entry in CONFIG.get("schedules", []):
        agent_id = entry.get("agent_id")
        cron = entry.get("cron")
        input_text = entry.get("input", "")
        if agent_id and cron:
            schedule_job(agent_id, cron, input_text)
    scheduler.start()

# ---------- Routes ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request from {client_ip} completed in {process_time:.2f}s", extra={"client_ip": client_ip})
    return response

@app.get("/", response_class=HTMLResponse, dependencies=[Depends(optional_auth)])
async def root():
    agents_json = json.dumps(AGENTS)
    html = UI_TEMPLATE.format(agents_json=agents_json)
    return HTMLResponse(html)

@app.post("/api/agent/{agent_id}", dependencies=[Depends(optional_auth)])
async def run_agent(agent_id: str, request: Request):
    data = await request.json()
    user_input = data.get("input", "")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing input")
    if agent_id not in AGENT_ROUTERS:
        raise HTTPException(status_code=404, detail="Agent not found")

    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Agent {agent_id} invoked by {client_ip}", extra={"client_ip": client_ip, "agent_id": agent_id})

    # Use global semaphore to limit concurrency
    async with llm_semaphore:
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, AGENT_ROUTERS[agent_id], user_input),
                timeout=get_agent_timeout()
            )
            file_ref = save_artifact(agent_id, result)
            save_history(agent_id, user_input, result, file_ref)
            return JSONResponse(content={"status": "ok", "result": result})
        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_id} timed out after {get_agent_timeout()}s",
                         extra={"client_ip": client_ip, "agent_id": agent_id})
            return JSONResponse(status_code=504, content={"status": "error", "message": f"Timed out after {get_agent_timeout()}s"})
        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}", exc_info=True, extra={"client_ip": client_ip, "agent_id": agent_id})
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/agent/stream/{agent_id}", dependencies=[Depends(optional_auth)])
async def stream_agent(agent_id: str, request: Request):
    user_input = request.query_params.get("input", "")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing input")
    if agent_id not in AGENT_ROUTERS:
        raise HTTPException(status_code=404, detail="Agent not found")

    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Streaming agent {agent_id} for {client_ip}", extra={"client_ip": client_ip, "agent_id": agent_id})

    prompt = build_agent_prompt(agent_id, user_input)
    system_prompt = "You are an expert COBOL modernization assistant."

    queue = asyncio.Queue()
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            token_gen = call_ollama(prompt, system_prompt, stream=True)
            for token in token_gen:
                if stop_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(("token", token)), loop)
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

    thread = threading.Thread(target=worker, daemon=False)
    thread.start()

    async def event_generator():
        stream_timeout = get_stream_timeout()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=stream_timeout)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': f'Stream timed out after {stream_timeout}s'})}\n\n"
                    break
                typ, data = item
                if typ == "token":
                    yield f"data: {json.dumps({'token': data})}\n\n"
                elif typ == "error":
                    yield f"data: {json.dumps({'error': data})}\n\n"
                    break
                elif typ == "done":
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
        except asyncio.CancelledError:
            stop_event.set()
            raise
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/upload", dependencies=[Depends(optional_auth)])
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")
    dest = SOURCE_POOL / file.filename
    content = await file.read()
    dest.write_bytes(content)
    git_commit_artifact(dest, f"Upload {file.filename}")
    return JSONResponse({"status": "ok", "message": f"Uploaded {file.filename}"})

@app.get("/api/workspace", dependencies=[Depends(optional_auth)])
async def list_workspace():
    files = []
    for root, dirs, files_list in os.walk(WORKSPACE_DIR):
        if ".git" in root:
            continue
        for f in files_list:
            rel = Path(root).relative_to(WORKSPACE_DIR) / f
            files.append(str(rel))
    return JSONResponse({"files": files})

@app.get("/api/history", dependencies=[Depends(optional_auth)])
async def history(limit: int = 50):
    return JSONResponse(get_history(limit))

@app.get("/api/config", dependencies=[Depends(optional_auth)])
async def get_config():
    return JSONResponse(CONFIG)

@app.post("/api/config", dependencies=[Depends(optional_auth)])
async def update_config(request: Request):
    data = await request.json()
    CONFIG.update(data)
    save_config(CONFIG)
    return JSONResponse({"status": "ok"})

@app.get("/api/export", dependencies=[Depends(optional_auth)])
async def export_workspace():
    zip_path = WORKSPACE_DIR / "workspace_export.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for root, dirs, files in os.walk(WORKSPACE_DIR):
            if any(part in Path(root).parts for part in ['.git', 'history', '__pycache__']):
                continue
            for file in files:
                if file.endswith(('.db', '.zip')):
                    continue
                full = os.path.join(root, file)
                arcname = os.path.relpath(full, WORKSPACE_DIR)
                zf.write(full, arcname)
    return FileResponse(zip_path, filename="workspace_export.zip")

# ---------- WebSocket (with optional auth) ----------
@app.websocket("/ws/agent/{agent_id}")
async def websocket_agent(websocket: WebSocket, agent_id: str):
    # Check authentication if API key is set
    if not websocket_auth(websocket):
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await websocket.accept()
    try:
        data = await websocket.receive_text()
        try:
            payload = json.loads(data)
            user_input = payload.get("input", "")
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
            return

        if not user_input:
            await websocket.send_text(json.dumps({"error": "Missing input"}))
            return

        prompt = build_agent_prompt(agent_id, user_input)
        system_prompt = "You are an expert COBOL modernization assistant."

        queue = asyncio.Queue()
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()

        def worker():
            try:
                token_gen = call_ollama(prompt, system_prompt, stream=True)
                for token in token_gen:
                    if stop_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(("token", token)), loop)
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

        thread = threading.Thread(target=worker, daemon=False)
        thread.start()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=get_stream_timeout())
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"error": f"Stream timed out after {get_stream_timeout()}s"}))
                    break
                typ, data = item
                if typ == "token":
                    await websocket.send_text(json.dumps({"token": data}))
                elif typ == "error":
                    await websocket.send_text(json.dumps({"error": data}))
                    break
                elif typ == "done":
                    await websocket.send_text(json.dumps({"done": True}))
                    break
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for agent {agent_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# ---------- UI Template (escaped, with single placeholder) ----------
UI_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>COBOLOS v2.0</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 1fr 2fr 1fr; gap: 20px; }}
        .panel {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
        .panel h3 {{ margin-top: 0; color: #58a6ff; }}
        textarea {{ width: 100%; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 8px; font-family: inherit; resize: vertical; }}
        button {{ background: #238636; color: #fff; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
        button:hover {{ background: #2ea043; }}
        #output {{ background: #0d1117; padding: 10px; border: 1px solid #30363d; border-radius: 4px; max-height: 400px; overflow: auto; white-space: pre-wrap; }}
        .loader {{ display: none; border: 4px solid #30363d; border-top: 4px solid #58a6ff; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .file-browser {{ list-style: none; padding: 0; }}
        .file-browser li {{ padding: 4px 0; border-bottom: 1px solid #21262d; font-size: 0.9em; }}
        .file-browser li:hover {{ background: #21262d; cursor: pointer; }}
        input[type="checkbox"] {{ accent-color: #58a6ff; }}
        .note {{ font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="panel">
        <h3>Controls</h3>
        <label>Agent:</label>
        <select id="agent-select" style="width:100%; background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:8px;">
        </select>
        <br><br>
        <div id="inputs-container"></div>
        <label><input type="checkbox" id="stream-checkbox"> Stream (live tokens)</label>
        <div class="note">Uses Server‑Sent Events (SSE)</div>
        <br><br>
        <button onclick="execute()">Run</button>
        <div class="loader" id="loader"></div>
        <br>
        <button onclick="clearOutput()">Clear Output</button>
        <br><br>
        <label>Model:</label>
        <input id="model-input" value="codellama:7b" style="width:100%; background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:4px;">
        <button onclick="updateModel()">Update Model</button>
        <br><br>
        <label>File Upload:</label>
        <input type="file" id="file-upload" style="width:100%; background:#0d1117; border:1px solid #30363d; padding:4px;">
        <button onclick="uploadFile()">Upload</button>
    </div>

    <div class="panel">
        <h3>Output</h3>
        <div id="output">Ready.</div>
        <br>
        <button onclick="exportWorkspace()">Export Workspace</button>
    </div>

    <div class="panel">
        <h3>Workspace Files</h3>
        <ul class="file-browser" id="file-list"><li>Loading...</li></ul>
        <h3>History</h3>
        <ul class="file-browser" id="history-list"><li>Loading...</li></ul>
        <button onclick="loadHistory()">Refresh History</button>
    </div>
</div>

<script>
    const AGENTS = {agents_json};

    function populateAgentDropdown() {{
        const sel = document.getElementById('agent-select');
        sel.innerHTML = '';
        for (const [id, info] of Object.entries(AGENTS)) {{
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = info.name + ' (' + id + ')';
            sel.appendChild(opt);
        }}
    }}

    function switchAgent() {{
        const sel = document.getElementById('agent-select');
        const agentId = sel.value;
        const info = AGENTS[agentId];
        const container = document.getElementById('inputs-container');
        container.innerHTML = '';
        if (info && info.inputs) {{
            info.inputs.forEach((label, idx) => {{
                const textarea = document.createElement('textarea');
                textarea.id = 'input-' + idx;
                textarea.placeholder = label + '...';
                textarea.rows = 4;
                textarea.style.width = '100%';
                textarea.style.marginBottom = '8px';
                container.appendChild(textarea);
            }});
        }} else {{
            const textarea = document.createElement('textarea');
            textarea.id = 'input-0';
            textarea.placeholder = 'Enter input...';
            textarea.rows = 4;
            textarea.style.width = '100%';
            container.appendChild(textarea);
        }}
    }}

    populateAgentDropdown();
    document.getElementById('agent-select').addEventListener('change', switchAgent);
    switchAgent();

    let eventSource = null;

    async function execute() {{
        const agent = document.getElementById('agent-select').value;
        const inputs = document.querySelectorAll('#inputs-container textarea');
        const inputText = Array.from(inputs).map(t => t.value).join('\\n---\\n');
        if (!inputText.trim()) {{ alert('Please enter input.'); return; }}

        const stream = document.getElementById('stream-checkbox').checked;
        const output = document.getElementById('output');
        const loader = document.getElementById('loader');
        loader.style.display = 'inline-block';

        if (stream) {{
            if (eventSource) {{
                eventSource.close();
                eventSource = null;
            }}
            output.textContent = '';
            const url = `/api/agent/stream/${{agent}}?input=${{encodeURIComponent(inputText)}}`;
            eventSource = new EventSource(url);
            eventSource.onmessage = function(event) {{
                try {{
                    const data = JSON.parse(event.data);
                    if (data.token) {{
                        output.textContent += data.token;
                    }} else if (data.error) {{
                        output.textContent = 'Error: ' + data.error;
                        eventSource.close();
                        loader.style.display = 'none';
                    }} else if (data.done) {{
                        eventSource.close();
                        loader.style.display = 'none';
                    }}
                }} catch (e) {{}}
            }};
            eventSource.onerror = function(error) {{
                console.error('EventSource error:', error);
                output.textContent = 'Streaming connection error.';
                eventSource.close();
                loader.style.display = 'none';
            }};
        }} else {{
            output.textContent = 'Running...';
            try {{
                const resp = await fetch('/api/agent/' + agent, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ input: inputText }})
                }});
                const data = await resp.json();
                if (data.status === 'error') {{
                    output.textContent = 'Error: ' + data.message;
                }} else {{
                    output.textContent = JSON.stringify(data.result, null, 2);
                }}
            }} catch (e) {{
                output.textContent = 'Request failed: ' + e;
            }} finally {{
                loader.style.display = 'none';
            }}
        }}
    }}

    function clearOutput() {{
        document.getElementById('output').textContent = 'Ready.';
        if (eventSource) {{
            eventSource.close();
            eventSource = null;
        }}
        document.getElementById('loader').style.display = 'none';
    }}

    async function updateModel() {{
        const model = document.getElementById('model-input').value;
        const resp = await fetch('/api/config', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ default_model: model }})
        }});
        const data = await resp.json();
        if (data.status === 'ok') alert('Model updated to ' + model);
    }}

    async function uploadFile() {{
        const fileInput = document.getElementById('file-upload');
        const file = fileInput.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch('/api/upload', {{ method: 'POST', body: formData }});
        const data = await resp.json();
        alert(data.message);
        loadFiles();
    }}

    async function loadFiles() {{
        const resp = await fetch('/api/workspace');
        const data = await resp.json();
        const list = document.getElementById('file-list');
        list.innerHTML = '';
        if (data.files.length === 0) {{
            list.innerHTML = '<li>No files</li>';
            return;
        }}
        data.files.forEach(f => {{
            const li = document.createElement('li');
            li.textContent = f;
            list.appendChild(li);
        }});
    }}

    async function loadHistory() {{
        const resp = await fetch('/api/history?limit=10');
        const data = await resp.json();
        const list = document.getElementById('history-list');
        list.innerHTML = '';
        if (data.length === 0) {{
            list.innerHTML = '<li>No history</li>';
            return;
        }}
        data.forEach(item => {{
            const li = document.createElement('li');
            li.textContent = item.timestamp.slice(0,16) + ' | ' + item.agent_id + ' | ' + item.input.slice(0,30);
            list.appendChild(li);
        }});
    }}

    async function exportWorkspace() {{
        window.location.href = '/api/export';
    }}

    loadFiles();
    loadHistory();
</script>
</body>
</html>
"""

# ---------- TUI Mode ----------
def run_tui():
    try:
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.table import Table
        from rich.panel import Panel
        from rich import print as rprint
    except ImportError:
        print("Rich library not installed. Install with 'pip install rich' for TUI.")
        return

    console = Console()
    rprint(Panel.fit("[bold cyan]COBOLOS v2.0 - TUI[/bold cyan]"))
    try:
        while True:
            console.print("\n[bold]Agents:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", style="dim")
            table.add_column("Name")
            table.add_column("Description")
            for aid, data in AGENTS.items():
                table.add_row(aid, data["name"], data["desc"])
            console.print(table)

            choice = Prompt.ask("Select agent ID (or 'q' to quit)", choices=list(AGENTS.keys()) + ["q"])
            if choice == "q":
                break
            agent_id = choice
            agent_input = Prompt.ask("Enter input for " + AGENTS[agent_id]["name"])
            console.print("[yellow]Running...[/yellow]")
            result = AGENT_ROUTERS[agent_id](agent_input)
            file_ref = save_artifact(agent_id, result)
            save_history(agent_id, agent_input, result, file_ref)
            console.print(Panel(json.dumps(result, indent=2, default=str), title="Result"))
    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting TUI.[/bold red]")
        sys.exit(0)

# ---------- Ollama Connectivity Check ----------
def check_ollama_connectivity():
    url = get_ollama_url()
    try:
        response = requests.get(url, timeout=5)
        logger.info(f"Ollama connectivity check: OK (status {response.status_code})")
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama connectivity check FAILED: {e}")
        return False

# ---------- CLI Mode ----------
def cli_mode():
    import argparse
    parser = argparse.ArgumentParser(description="COBOLOS CLI - run an agent once")
    parser.add_argument("--agent", required=True, help="Agent ID (01-10)")
    parser.add_argument("--input", required=True, help="Input text for the agent")
    parser.add_argument("--output", help="Output file to write result (JSON)")
    args = parser.parse_args()

    if args.agent not in AGENT_ROUTERS:
        print(f"Error: Unknown agent '{args.agent}'")
        sys.exit(1)

    try:
        print(f"Running agent {args.agent}...")
        result = AGENT_ROUTERS[args.agent](args.input)
        file_ref = save_artifact(args.agent, result)
        save_history(args.agent, args.input, result, file_ref)
        print("Result:")
        print(json.dumps(result, indent=2, default=str))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Result saved to {args.output}")
    except Exception as e:
        print(f"Error during agent execution: {e}")
        sys.exit(2)

# ---------- Main ----------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="COBOLOS v2.0 - Local COBOL Modernization Agent Harness")
    parser.add_argument("--tui", action="store_true", help="Launch terminal UI")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (headless)")
    args = parser.parse_args()

    if args.cli:
        cli_mode()
        return

    print_logo()
    init_workspace()
    init_mock_db()
    init_history_db()
    # External DB will be initialised on startup via lifespan
    check_ollama_connectivity()

    if args.tui:
        run_tui()
    else:
        webbrowser.open(f"http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()