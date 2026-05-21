#!/usr/bin/env python3
"""
STACK – Local Repository Agent with nomic-embed-text
Auto‑bootstraps venv + dependencies on first run.
"""

import os
import sys
import json
import math
import time
import threading
import hashlib
import subprocess
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# 0. VENV BOOTSTRAP – prevent “externally-managed” errors
# ============================================================================
STACK_DIR = Path(__file__).resolve().parent
VENV_DIR = STACK_DIR / ".stack_venv"

def is_venv() -> bool:
    """Check if we are already running inside a virtual environment."""
    return (
        hasattr(sys, 'real_prefix')
        or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

def create_venv():
    """Create the virtual environment and return the path to its Python binary."""
    print("[*] Creating virtual environment...")
    venv.create(str(VENV_DIR), with_pip=True)
    if os.name == 'nt':
        python = VENV_DIR / 'Scripts' / 'python.exe'
    else:
        python = VENV_DIR / 'bin' / 'python'
    return str(python)

def install_dependencies(python_exe: str):
    """Install required packages inside the virtual environment."""
    packages = ["ollama"]
    print(f"[*] Installing dependencies inside venv: {packages}")
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([python_exe, "-m", "pip", "install"] + packages)

def relaunch_in_venv(python_exe: str):
    """Replace the current process with one running inside the venv."""
    print("[*] Relaunching inside virtual environment...")
    os.execv(python_exe, [python_exe, __file__] + sys.argv[1:])

# Bootstrap logic
if not is_venv():
    print("[!] Not running in a virtual environment.")
    if VENV_DIR.exists():
        # Venv exists – use it
        if os.name == 'nt':
            python_exe = str(VENV_DIR / 'Scripts' / 'python.exe')
        else:
            python_exe = str(VENV_DIR / 'bin' / 'python')
        if not Path(python_exe).exists():
            print("[!] Venv appears corrupt, recreating...")
            python_exe = create_venv()
            install_dependencies(python_exe)
        relaunch_in_venv(python_exe)
    else:
        # First run – create venv, install deps, relaunch
        python_exe = create_venv()
        install_dependencies(python_exe)
        relaunch_in_venv(python_exe)

# ============================================================================
# Now safely inside the venv – import ollama
# ============================================================================
import ollama

# ============================================================================
# 2. CONFIGURATION & PATHS
# ============================================================================
STACK_ROOT = os.path.abspath("./stack_system")
WORKSPACE = os.path.join(STACK_ROOT, "current_workspace")
TEMPLATES_DIR = os.path.join(STACK_ROOT, "templates")
MANIFEST_PATH = os.path.join(TEMPLATES_DIR, "manifest.json")
MEMORY_FILE = os.path.join(STACK_ROOT, ".stack_memory.json")
MAX_TOOL_DEPTH = 10

os.makedirs(WORKSPACE, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

_manifest_lock = threading.RLock()

# ============================================================================
# 3. OLLAMA CONNECTION & MODEL MANAGEMENT
# ============================================================================
def check_ollama_running() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False

def start_ollama() -> bool:
    try:
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        time.sleep(3)
        return check_ollama_running()
    except Exception:
        return False

def ensure_models() -> None:
    print("[*] Verifying Ollama models...")
    if not check_ollama_running():
        print("[!] Ollama not running. Attempting to start...")
        if not start_ollama():
            print("[!] Could not start Ollama. Please run 'ollama serve' manually.")
            sys.exit(1)

    required_models = ["nomic-embed-text"]
    try:
        installed = ollama.list()
        installed_names = [m.model for m in installed.models] if hasattr(installed, 'models') else []
    except Exception:
        installed_names = []

    for model in required_models:
        if model not in installed_names:
            print(f"[*] Pulling {model}...")
            try:
                ollama.pull(model)
                print(f"[✓] {model} ready")
            except Exception as e:
                print(f"[!] Failed to pull {model}: {e}")
                sys.exit(1)

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        response = ollama.embed(model="nomic-embed-text", input=text)
        if hasattr(response, 'embeddings'):
            embeddings = response.embeddings
        elif isinstance(response, dict):
            embeddings = response.get('embeddings', [])
        else:
            return None
        if embeddings and len(embeddings) > 0:
            return embeddings[0] if isinstance(embeddings[0], list) else embeddings[0]
        return None
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return None

# ============================================================================
# 4. VECTOR SIMILARITY & MEMORY SYSTEM
# ============================================================================
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def remember_action(action_summary: str, metadata: Dict = None) -> None:
    memory = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                memory = json.load(f)
        except json.JSONDecodeError:
            memory = []
    vector = get_embedding(action_summary)
    if vector:
        entry = {
            "text": action_summary,
            "vector": vector,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        memory.append(entry)
        if len(memory) > 1000:
            memory = memory[-1000:]
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f)

def query_memory(query: str, top_k: int = 3) -> List[str]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r') as f:
            memory = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    query_vector = get_embedding(query)
    if not query_vector:
        return []
    scored = []
    for item in memory:
        if "vector" in item and item["vector"]:
            score = cosine_similarity(query_vector, item["vector"])
            scored.append((score, item["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for score, text in scored[:top_k] if score > 0.3]

# ============================================================================
# 5. TEMPLATE MANAGEMENT & HOT RELOAD
# ============================================================================
DEFAULT_TEMPLATES = {
    "python_web_server": {
        "description": "minimal fastapi python web server with api routes",
        "files": {
            "app.py": "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef root():\n    return {'status': 'STACK running'}\n",
            "requirements.txt": "fastapi\nuvicorn"
        }
    },
    "html_login_component": {
        "description": "modern tailwind css login form ui component",
        "files": {
            "login.html": """<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center h-screen">
    <div class="bg-white p-8 rounded-lg shadow-md w-96">
        <h2 class="text-2xl font-bold mb-6">Login</h2>
        <input type="email" placeholder="Email" class="w-full p-2 border rounded mb-4">
        <input type="password" placeholder="Password" class="w-full p-2 border rounded mb-4">
        <button class="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600">Sign In</button>
    </div>
</body>
</html>"""
        }
    }
}

def seed_initial_templates() -> None:
    if os.path.exists(MANIFEST_PATH):
        return
    print("[*] Seeding initial templates...")
    manifest = {}
    for name, data in DEFAULT_TEMPLATES.items():
        vector = get_embedding(data["description"])
        if vector:
            manifest[name] = {
                "description": data["description"],
                "vector": vector,
                "files": data["files"]
            }
            print(f"  ✓ {name}")
    with _manifest_lock:
        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)

def scan_templates_folder() -> Dict:
    discovered = {}
    if not os.path.exists(TEMPLATES_DIR):
        return discovered
    for item in os.listdir(TEMPLATES_DIR):
        item_path = os.path.join(TEMPLATES_DIR, item)
        if not os.path.isdir(item_path):
            continue
        files_dict = {}
        description = f"code template for {item.replace('_', ' ')}"
        for root, _, files in os.walk(item_path):
            for filename in files:
                if filename == "description.txt":
                    try:
                        with open(os.path.join(root, filename), 'r') as f:
                            description = f.read().strip()
                    except Exception:
                        pass
                    continue
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, item_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files_dict[rel_path] = f.read()
                except Exception:
                    pass
        if files_dict:
            discovered[item] = {
                "description": description,
                "files": files_dict
            }
    return discovered

def reload_manifest() -> None:
    raw_templates = scan_templates_folder()
    with _manifest_lock:
        manifest = {}
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, 'r') as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                manifest = {}
        updated = False
        for name, data in raw_templates.items():
            if name not in manifest or manifest[name]["description"] != data["description"]:
                vector = get_embedding(data["description"])
                if vector:
                    manifest[name] = {
                        "description": data["description"],
                        "vector": vector,
                        "files": data["files"]
                    }
                    updated = True
                    print(f"  ✓ Updated template: {name}")
        if updated:
            with open(MANIFEST_PATH, 'w') as f:
                json.dump(manifest, f, indent=2)

def start_template_watcher() -> threading.Thread:
    def watcher():
        last_mtime = {}
        while True:
            time.sleep(2)
            try:
                current = {}
                if os.path.exists(TEMPLATES_DIR):
                    for root, _, files in os.walk(TEMPLATES_DIR):
                        for f in files:
                            path = os.path.join(root, f)
                            current[path] = os.path.getmtime(path)
                if current != last_mtime:
                    reload_manifest()
                    last_mtime = current
            except Exception:
                pass
    thread = threading.Thread(target=watcher, daemon=True)
    thread.start()
    return thread

# ============================================================================
# 6. SANDBOXED GIT & FILE OPERATIONS
# ============================================================================
def validate_workspace_path(target_path: str) -> bool:
    abs_target = os.path.abspath(os.path.join(WORKSPACE, target_path))
    abs_workspace = os.path.abspath(WORKSPACE)
    return abs_target.startswith(abs_workspace)

def run_git_safe(args: List[str], error_ok: bool = False) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout or "OK"
        elif error_ok:
            return False, result.stderr
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Git operation timed out"
    except Exception as e:
        return False, str(e)

def cmd_build(repo_name: str) -> str:
    global WORKSPACE
    new_workspace = os.path.join(STACK_ROOT, repo_name)
    try:
        os.makedirs(new_workspace, exist_ok=True)
        WORKSPACE = new_workspace
        success, msg = run_git_safe(["init"])
        if not success:
            return f"Git init failed: {msg}"
        success, msg = run_git_safe(["checkout", "-b", "main"])
        config_path = os.path.join(WORKSPACE, "STACK.json")
        with open(config_path, 'w') as f:
            json.dump({
                "name": repo_name,
                "created": time.time(),
                "managed_by": "STACK"
            }, f)
        run_git_safe(["add", "."])
        run_git_safe(["commit", "-m", "chore: initial STACK bootstrap"])
        remember_action(f"Created repository '{repo_name}'", {"action": "build", "repo": repo_name})
        return f"✓ Repository '{repo_name}' created at {WORKSPACE}"
    except Exception as e:
        return f"✗ Build failed: {e}"

def cmd_add(search_query: str) -> str:
    workspace_git = os.path.join(WORKSPACE, ".git")
    if not os.path.exists(workspace_git):
        return "✗ No active repository. Use '/build <name>' first"
    with _manifest_lock:
        if not os.path.exists(MANIFEST_PATH):
            return "✗ No templates available. Run '/import' or add to templates folder"
        try:
            with open(MANIFEST_PATH, 'r') as f:
                manifest = json.load(f)
        except Exception:
            return "✗ Corrupted manifest file"
    if not manifest:
        return "✗ Manifest is empty. Seed templates first"
    query_vector = get_embedding(search_query)
    if not query_vector:
        return "✗ Could not generate embedding for query"
    best_match = None
    best_score = -1.0
    for name, data in manifest.items():
        if "vector" in data:
            score = cosine_similarity(query_vector, data["vector"])
            if score > best_score:
                best_score = score
                best_match = (name, data["files"], data["description"])
    if best_score < 0.35:
        return f"✗ No good match (confidence: {best_score:.2f}). Try different phrasing"
    template_name, files, description = best_match
    branch_name = f"stack/{template_name}"
    success, msg = run_git_safe(["checkout", "-b", branch_name], error_ok=True)
    files_written = []
    for filename, content in files.items():
        target = os.path.join(WORKSPACE, filename)
        if not validate_workspace_path(filename):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        files_written.append(filename)
    run_git_safe(["add", "."])
    commit_msg = f"feat: add {template_name} via STACK\n\n{description}"
    run_git_safe(["commit", "-m", commit_msg])
    run_git_safe(["checkout", "main"])
    run_git_safe(["merge", branch_name])
    remember_action(f"Added template '{template_name}' to repository",
                    {"action": "add", "template": template_name, "files": files_written})
    return f"✓ Added '{template_name}' (confidence: {best_score:.2f})\n  Files: {', '.join(files_written)}"

def cmd_import(json_path: str) -> str:
    if not os.path.exists(json_path):
        return f"✗ File not found: {json_path}"
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            new_templates = json.load(f)
    except Exception as e:
        return f"✗ Invalid JSON: {e}"
    with _manifest_lock:
        manifest = {}
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, 'r') as f:
                    manifest = json.load(f)
            except Exception:
                pass
        imported = 0
        for name, data in new_templates.items():
            if "description" not in data or "files" not in data:
                continue
            vector = get_embedding(data["description"])
            if vector:
                manifest[name] = {
                    "description": data["description"],
                    "vector": vector,
                    "files": data["files"]
                }
                imported += 1
        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)
    return f"✓ Imported {imported} templates from {json_path}"

def cmd_status() -> str:
    with _manifest_lock:
        manifest_count = 0
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, 'r') as f:
                    manifest_count = len(json.load(f))
            except Exception:
                pass
    has_git = os.path.exists(os.path.join(WORKSPACE, ".git"))
    lines = [
        f"STACK Status",
        f"  Workspace: {WORKSPACE}",
        f"  Git initialized: {'✓' if has_git else '✗'}",
        f"  Templates in manifest: {manifest_count}",
        f"  Ollama: {'✓' if check_ollama_running() else '✗'}"
    ]
    return "\n".join(lines)

def cmd_list_templates() -> str:
    with _manifest_lock:
        if not os.path.exists(MANIFEST_PATH):
            return "No templates found"
        try:
            with open(MANIFEST_PATH, 'r') as f:
                manifest = json.load(f)
        except Exception:
            return "Error reading manifest"
    if not manifest:
        return "No templates in manifest"
    lines = ["Available templates:"]
    for name, data in manifest.items():
        desc = data.get("description", "No description")[:60]
        lines.append(f"  • {name}: {desc}")
    return "\n".join(lines)

# ============================================================================
# 7. CLI INTERFACE
# ============================================================================
def print_banner() -> None:
    banner = """
  ███████╗████████╗ █████╗  ██████╗██╗  ██╗
  ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
  ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
  ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
  ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
        Embed-Driven Repository Architect
"""
    print(banner)

def print_commands() -> None:
    print("""
Commands:
  /build <name>        - Create new git repository workspace
  /add <description>   - Find matching template, inject into workspace
  /import <file.json>  - Import external template bundle
  /list                - List available templates
  /status              - Show system status
  /help                - Show this help
  /quit                - Exit STACK

Templates folder: ./stack_system/templates/
  - Add folders with code + description.txt for hot-reload
  - JSON imports also supported
""")

def main() -> None:
    print("[*] Initializing STACK...")
    ensure_models()
    seed_initial_templates()
    start_template_watcher()
    print_banner()
    print_commands()
    while True:
        try:
            user_input = input("\nSTACK > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                print("[*] Goodbye!")
                break
            if user_input == "/help":
                print_commands()
                continue
            if user_input == "/status":
                print(cmd_status())
                continue
            if user_input == "/list":
                print(cmd_list_templates())
                continue
            if user_input.startswith("/build "):
                repo_name = user_input[7:].strip()
                if repo_name:
                    print(cmd_build(repo_name))
                else:
                    print("Usage: /build <repo_name>")
                continue
            if user_input.startswith("/add "):
                query = user_input[5:].strip()
                if query:
                    print(cmd_add(query))
                else:
                    print("Usage: /add <description>")
                continue
            if user_input.startswith("/import "):
                path = user_input[8:].strip()
                if path:
                    print(cmd_import(path))
                else:
                    print("Usage: /import <path/to/template.json>")
                continue
            print(f"Unknown command. Type /help for available commands.")
        except KeyboardInterrupt:
            print("\n[*] Interrupted. Goodbye!")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
