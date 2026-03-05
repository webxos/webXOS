#!/usr/bin/env python3
"""
PygmyClaw – Compact AI Agent with multi‑instance speculative decoding and persistent queue.
"""
import os
import json
import sys
import subprocess
import urllib.request
import urllib.error
import socket
import time
import threading
import queue
from pathlib import Path

# Optional Redis support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Optional Ollama Python client (for multi‑instance drafts)
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "config.json"
ERROR_LOG = None
MAX_LOG_ENTRIES = 1000

DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEBUG = os.environ.get("PYGMYCLAW_DEBUG", "").lower() in ("1", "true", "yes")

# ----------------------------------------------------------------------
# Multi‑instance / speculative decoding globals
INSTANCE_PROCESSES = []          # list of Popen objects for each Ollama serve
DRAFT_BATCH_SIZE = 6              # tokens per draft
USE_REDIS = False                 # set by config
REDIS_CLIENT = None
QUEUE_NAME = "grok_tasks"
TASK_QUEUE = None                 # fallback Python queue
QUEUE_PROCESSOR_EVENT = threading.Event()

# ----------------------------------------------------------------------
class PygmyClaw:
    def __init__(self):
        self.workspace = SCRIPT_DIR
        self.model = DEFAULT_MODEL
        self.endpoint = DEFAULT_ENDPOINT
        self.multi_instance = None
        self.queue_config = None
        self.load_config()
        self.multitool = self.workspace / "pygmyclaw_multitool.py"
        self.check_prerequisites()
        self._ensure_model_ready()
        self._warmup_model()

        # Fetch Python tool list
        py_resp = self.call_multitool("list_tools_detailed")
        self.python_tools = self._extract_tool_list(py_resp)

        self.system_prompt = self.build_system_prompt()

        # Initialize queue if multi‑instance is enabled
        if self.multi_instance and self.multi_instance.get("enabled"):
            self._init_queue()
            self.start_queue_processor()

    def _extract_tool_list(self, resp):
        """Extract list of tools from JSON response."""
        if not isinstance(resp, dict):
            return []
        if "error" in resp:
            print(f"⚠️ Tool list error: {resp['error']}", file=sys.stderr)
            return []
        if "result" in resp and isinstance(resp["result"], dict):
            inner = resp["result"]
            if "tools" in inner and isinstance(inner["tools"], list):
                return inner["tools"]
        if "tools" in resp and isinstance(resp["tools"], list):
            return resp["tools"]
        return []

    def load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    cfg = json.load(f)
                    self.model = cfg.get("model", self.model)
                    self.endpoint = cfg.get("endpoint", self.endpoint)
                    if "workspace" in cfg:
                        self.workspace = Path(cfg["workspace"]).resolve()
                    if cfg.get("debug", False):
                        global DEBUG
                        DEBUG = True
                    self.multi_instance = cfg.get("multi_instance")
                    self.queue_config = cfg.get("queue")
            except Exception as e:
                print(f"⚠️ Warning: Could not load config.json: {e}")
        global ERROR_LOG
        ERROR_LOG = self.workspace / "error_log.json"

    def check_prerequisites(self):
        if not self.multitool.exists():
            print(f"❌ Python multitool not found at {self.multitool}")
            sys.exit(1)

        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if self.model not in models and not any(m.startswith(self.model) for m in models):
                    print(f"⚠️ Model '{self.model}' not found in local list.")
                else:
                    print(f"✅ Model '{self.model}' found locally.")
        except Exception as e:
            print(f"❌ Cannot reach Ollama at {self.endpoint}: {e}")
            print("   Make sure Ollama is running (try 'ollama serve' in another terminal).")
            sys.exit(1)

    def _ensure_model_ready(self):
        print(f"⏳ Ensuring model '{self.model}' is ready...")
        test_payload = {
            "model": self.model,
            "prompt": "hello",
            "stream": False,
            "options": {"num_predict": 1}
        }
        try:
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(test_payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_data = json.loads(resp.read())
                if "response" in resp_data:
                    print("✅ Model is ready.")
                else:
                    print("⚠️ Unexpected response from Ollama.")
        except Exception as e:
            print(f"❌ Failed to communicate with model '{self.model}': {e}")
            print("\nPossible solutions:")
            print("1. Ensure Ollama is running: `ollama serve`")
            print("2. Pull the model manually: `ollama pull {}`".format(self.model))
            sys.exit(1)

    def _warmup_model(self):
        try:
            warmup = {
                "model": self.model,
                "prompt": ".",
                "stream": False,
                "options": {"num_predict": 1}
            }
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(warmup).encode(),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Multi‑instance management
    def start_instances(self):
        """Launch 4 Ollama instances on different ports."""
        if not self.multi_instance or not self.multi_instance.get("enabled"):
            print("Multi‑instance not enabled in config.")
            return

        ports = self.multi_instance.get("ports", [11434, 11435, 11436, 11437])
        global INSTANCE_PROCESSES
        for i, port in enumerate(ports):
            env = os.environ.copy()
            env['OLLAMA_HOST'] = f'127.0.0.1:{port}'
            env['OLLAMA_NUM_PARALLEL'] = '1'
            if i > 0 and 'CUDA_VISIBLE_DEVICES' in env:
                gpu_ids = env['CUDA_VISIBLE_DEVICES'].split(',')
                if len(gpu_ids) > i:
                    env['CUDA_VISIBLE_DEVICES'] = gpu_ids[i]
                else:
                    env.pop('CUDA_VISIBLE_DEVICES', None)
            proc = subprocess.Popen(['ollama', 'serve'], env=env)
            INSTANCE_PROCESSES.append(proc)
            print(f"Started Ollama on port {port} (PID {proc.pid})")
            time.sleep(2)

    def stop_instances(self):
        global INSTANCE_PROCESSES
        for proc in INSTANCE_PROCESSES:
            proc.terminate()
        INSTANCE_PROCESSES.clear()
        print("All instances stopped.")

    # ------------------------------------------------------------------
    # Tokenization helper (uses Ollama's tokenize endpoint)
    def _tokenize(self, text):
        """Return list of token strings for the given text."""
        url = self.endpoint.replace("/generate", "/tokenize")
        payload = {"model": self.model, "prompt": text}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data.get("tokens", [])
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Tokenization failed: {e}", file=sys.stderr)
            return []

    def _count_tokens(self, text):
        return len(self._tokenize(text))

    # ------------------------------------------------------------------
    # Speculative decoding with 3 drafters + 1 verifier
    def _draft_gen_ollama(self, host, prompt, batch_size):
        if not OLLAMA_CLIENT_AVAILABLE:
            raise RuntimeError("Ollama Python client not installed.")
        client = ollama.Client(host=host)
        resp = client.generate(
            model=self.model,
            prompt=prompt,
            options={'num_predict': batch_size, 'temperature': 0.6}
        )
        return resp['response']

    def generate_with_ssd(self, prompt, max_tokens=100):
        """
        Speculative decoding:
        - 3 drafters generate parallel drafts (each a string of batch_size tokens)
        - Verifier checks the last token of the longest draft
        """
        if not self.multi_instance or not self.multi_instance.get("enabled"):
            return self.ask_ollama(prompt)

        if not OLLAMA_CLIENT_AVAILABLE:
            print("⚠️ Ollama client not available – falling back to single instance.")
            return self.ask_ollama(prompt)

        ports = self.multi_instance.get("ports")
        if len(ports) < 4:
            raise RuntimeError("Need at least 4 ports for SSD")

        output = ""
        current_prompt = prompt
        tokens_generated = 0

        while tokens_generated < max_tokens:
            drafts = [None] * 3
            threads = []

            def draft_worker(idx):
                host = f'http://127.0.0.1:{ports[idx]}'
                try:
                    drafts[idx] = self._draft_gen_ollama(host, current_prompt, DRAFT_BATCH_SIZE)
                except Exception as e:
                    if DEBUG:
                        print(f"[DEBUG] Drafter {idx} failed: {e}", file=sys.stderr)

            for i in range(3):
                t = threading.Thread(target=draft_worker, args=(i,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # Filter out None drafts and pick longest
            valid_drafts = [d for d in drafts if d]
            if not valid_drafts:
                # Fallback to single instance if all drafters failed
                return self.ask_ollama(prompt)

            best_draft = max(valid_drafts, key=len)

            # Tokenize the best draft to get the last token
            draft_tokens = self._tokenize(best_draft)
            if not draft_tokens:
                # If tokenization fails, accept whole draft (conservative)
                accepted = best_draft
            else:
                last_draft_token = draft_tokens[-1]

                # Verifier generates one token given prompt + draft without last token
                verify_host = f'http://127.0.0.1:{ports[3]}'
                vclient = ollama.Client(host=verify_host)
                prompt_with_draft = current_prompt + best_draft[:-len(last_draft_token)]  # remove last token
                try:
                    vresp = vclient.generate(
                        model=self.model,
                        prompt=prompt_with_draft,
                        options={'num_predict': 1}
                    )
                    verifier_text = vresp['response']
                except Exception as e:
                    if DEBUG:
                        print(f"[DEBUG] Verifier failed: {e}", file=sys.stderr)
                    # Fallback: accept the whole draft
                    accepted = best_draft
                else:
                    verifier_tokens = self._tokenize(verifier_text)
                    if verifier_tokens and verifier_tokens[0] == last_draft_token:
                        accepted = best_draft
                    else:
                        accepted = verifier_text
                        if DEBUG:
                            print("Grok twist: Draft rejected – too boring!")

            output += accepted
            current_prompt += accepted
            tokens_generated += self._count_tokens(accepted)

        return output

    # ------------------------------------------------------------------
    # Persistent queue (Redis or file)
    def _init_queue(self):
        global USE_REDIS, REDIS_CLIENT, TASK_QUEUE, QUEUE_NAME
        qcfg = self.queue_config
        if qcfg and qcfg.get("type") == "redis" and REDIS_AVAILABLE:
            try:
                REDIS_CLIENT = redis.Redis(
                    host=qcfg.get("redis_host", "localhost"),
                    port=qcfg.get("redis_port", 6379),
                    db=0
                )
                REDIS_CLIENT.ping()
                USE_REDIS = True
                QUEUE_NAME = qcfg.get("queue_name", "grok_tasks")
                print("Using Redis queue.")
            except Exception as e:
                print(f"Redis unavailable, falling back to file queue: {e}")
                USE_REDIS = False
        else:
            USE_REDIS = False

        if not USE_REDIS:
            TASK_QUEUE = queue.Queue()
            qfile = self.workspace / "task_queue.json"
            if qfile.exists():
                try:
                    with open(qfile) as f:
                        tasks = json.load(f)
                        for t in tasks:
                            TASK_QUEUE.put(t)
                except Exception:
                    pass

    def _save_file_queue(self):
        if USE_REDIS:
            return
        qfile = self.workspace / "task_queue.json"
        tasks = list(TASK_QUEUE.queue)
        try:
            with open(qfile, 'w') as f:
                json.dump(tasks, f)
        except Exception:
            pass

    def add_task(self, prompt):
        task = {"id": str(time.time()), "prompt": prompt}
        if USE_REDIS:
            REDIS_CLIENT.rpush(QUEUE_NAME, json.dumps(task))
        else:
            TASK_QUEUE.put(task)
            self._save_file_queue()
        print(f"Task {task['id']} added to queue.")

    def process_queue(self):
        """Background worker: continuously process queued tasks."""
        print("Queue processor started.")
        while QUEUE_PROCESSOR_EVENT.is_set():
            task = None
            if USE_REDIS:
                data = REDIS_CLIENT.lpop(QUEUE_NAME)
                if data:
                    task = json.loads(data)
            else:
                try:
                    task = TASK_QUEUE.get(timeout=1)
                except queue.Empty:
                    pass

            if task:
                print(f"Processing task {task['id']}: {task['prompt']}")
                try:
                    result = self.generate_with_ssd(task['prompt'], max_tokens=150)
                    out_file = self.workspace / f"task_{task['id']}.out"
                    with open(out_file, 'w') as f:
                        f.write(result)
                    print(f"Task {task['id']} completed -> {out_file}")
                except Exception as e:
                    print(f"Task {task['id']} failed: {e}")
            else:
                time.sleep(1)

    def start_queue_processor(self):
        QUEUE_PROCESSOR_EVENT.set()
        thr = threading.Thread(target=self.process_queue, daemon=True)
        thr.start()

    def stop_queue_processor(self):
        QUEUE_PROCESSOR_EVENT.clear()

    # ------------------------------------------------------------------
    # Core methods for interacting with Python tools
    def call_multitool(self, action, **kwargs):
        payload = {"action": action, **kwargs}
        if DEBUG:
            print(f"[DEBUG] Calling Python tool: {action}", file=sys.stderr)
        try:
            proc = subprocess.run(
                [sys.executable, str(self.multitool)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=30
            )
            if proc.returncode != 0:
                return {"error": f"Python tool exited with code {proc.returncode}: {proc.stderr}"}
            return json.loads(proc.stdout)
        except subprocess.TimeoutExpired:
            return {"error": "Python tool timeout"}
        except json.JSONDecodeError:
            self.log_error("JSON decode error from Python tool", proc.stdout)
            return {"error": "Invalid JSON from Python tool"}
        except Exception as e:
            self.log_error(str(e))
            return {"error": str(e)}

    def log_error(self, msg, trace=""):
        self.call_multitool("log_error", msg=msg, trace=trace)

    def build_system_prompt(self):
        tools_desc = ""
        for t in self.python_tools:
            params = t.get("parameters", {})
            tools_desc += f"- {t['name']}: {t['description']} (parameters: {json.dumps(params)})\n"
        return f"""You are PygmyClaw, a compact AI assistant with access to these tools:
{tools_desc}

To use a tool, respond with a JSON object:
{{"tool": "tool_name", "parameters": {{"param1": "value", ...}}}}

Otherwise, respond normally with plain text.
"""

    def ask_ollama(self, user_input, system=None):
        payload = {
            "model": self.model,
            "prompt": user_input,
            "system": system if system else self.system_prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 512}
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        if DEBUG:
            print(f"[DEBUG] Ollama request: {payload}", file=sys.stderr)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                response_data = json.loads(resp.read())
                return response_data.get("response", "").strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            self.log_error(f"Ollama HTTP {e.code}", error_body)
            return f"Ollama error: {e.code} - {error_body}"
        except urllib.error.URLError as e:
            reason = str(e.reason) if hasattr(e, 'reason') else str(e)
            self.log_error(f"Ollama URL error: {reason}")
            return f"Error contacting Ollama (URL error): {reason}"
        except socket.timeout:
            self.log_error("Ollama timeout")
            return "Error: Ollama request timed out after 300 seconds."
        except Exception as e:
            self.log_error(str(e))
            return f"Error contacting Ollama: {e}"

    def extract_json(self, text):
        text = text.strip()
        start = text.find('{')
        if start == -1:
            return None
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        return None

    def run_tool_if_needed(self, user_input):
        print("PygmyClaw> Thinking...", end='', flush=True)
        response = self.ask_ollama(user_input)
        print("\r" + " "*30 + "\r", end='', flush=True)

        cmd = self.extract_json(response)
        if cmd and isinstance(cmd, dict):
            tool = cmd.get("tool")
            params = cmd.get("parameters", {})
            if tool and isinstance(params, dict):
                python_names = {t['name'] for t in self.python_tools}
                if tool in python_names:
                    result = self.call_multitool(tool, **params)
                else:
                    result = {"error": f"Unknown tool '{tool}'"}
                followup = (
                    f"User asked: {user_input}\n"
                    f"You used tool '{tool}' with result:\n{json.dumps(result, indent=2)}\n"
                    f"Now provide a helpful response to the user based on that result."
                )
                final = self.ask_ollama(followup, system=self.system_prompt)
                return final

        return response

    def repl(self):
        print(f"🐍 PygmyClaw (model: {self.model}, workspace: {self.workspace}) – Type /help for commands.")
        while True:
            try:
                user_input = input(">> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("/exit", "/q"):
                    break
                if user_input == "/help":
                    self.show_help()
                    continue
                response = self.run_tool_if_needed(user_input)
                print(f"Claw: {response}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log_error(str(e))
                print(f"Error: {e}")

    def show_help(self):
        all_tools = [t['name'] for t in self.python_tools]
        try:
            sysinfo = self.call_multitool("sys_info")
            os_name = sysinfo.get("os", "unknown")
        except Exception:
            os_name = "unknown"
        print("\n" + "="*50)
        print(f"🐍 PygmyClaw – {os_name}")
        print("="*50)
        print("Built‑in commands:")
        print("  /help            Show this menu")
        print("  /exit, /q        Quit")
        print("\nAvailable tools:")
        for t in sorted(all_tools):
            print(f"  - {t}")
        print("="*50)

# ----------------------------------------------------------------------
# CLI entry point with subcommands
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PygmyClaw with speculative decoding")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    sp = subparsers.add_parser("start", help="Start 4 Ollama instances")
    sp.add_argument("--background", action="store_true", help="Run queue processor in background")

    subparsers.add_parser("stop", help="Stop all instances")

    sp = subparsers.add_parser("generate", help="Generate text using speculative decoding")
    sp.add_argument("prompt", type=str, help="Input prompt")
    sp.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")

    sp = subparsers.add_parser("queue", help="Queue operations")
    sp.add_argument("action", choices=["add", "process", "status"], help="add a task, start processor, or show status")
    sp.add_argument("prompt", nargs="?", help="Prompt to add (for 'add')")

    subparsers.add_parser("repl", help="Start interactive REPL (default)")

    args = parser.parse_args()

    agent = PygmyClaw()

    if args.command == "start":
        agent.start_instances()
        if args.background:
            print("Queue processor is running (background).")
        else:
            print("Instances started. Use Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                agent.stop_instances()

    elif args.command == "stop":
        agent.stop_instances()
        agent.stop_queue_processor()

    elif args.command == "generate":
        result = agent.generate_with_ssd(args.prompt, max_tokens=args.max_tokens)
        print(result)

    elif args.command == "queue":
        if args.action == "add":
            if not args.prompt:
                print("Error: prompt required for 'add'")
                sys.exit(1)
            agent.add_task(args.prompt)
        elif args.action == "process":
            agent.process_queue()
        elif args.action == "status":
            if USE_REDIS:
                length = REDIS_CLIENT.llen(QUEUE_NAME)
                print(f"Redis queue '{QUEUE_NAME}' has {length} tasks.")
            else:
                length = TASK_QUEUE.qsize()
                print(f"In‑memory queue has {length} tasks.")
        else:
            print("Unknown queue action")

    else:
        agent.repl()

if __name__ == "__main__":
    main()
