#!/usr/bin/env python3
"""
PygmyClaw Multitool – Contains the actual tool implementations.
Now with generic dispatcher, heartbeat, file I/O, and scheduler tools.
"""
import json
import sys
import os
import time
import inspect
import platform
from pathlib import Path

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

SCRIPT_DIR = Path(__file__).parent.resolve()
ERROR_LOG = SCRIPT_DIR / "error_log.json"
MAX_LOG_ENTRIES = 1000
SCHEDULED_JOBS_FILE = SCRIPT_DIR / "scheduled_jobs.json"

# ----------------------------------------------------------------------
# Tool definitions
TOOLS = {
    "list_tools_detailed": {
        "name": "list_tools_detailed",
        "description": "List all available tools with their descriptions and parameters.",
        "parameters": {},
        "func": "do_list_tools"
    },
    "sys_info": {
        "name": "sys_info",
        "description": "Get system information (OS, Python version, etc.).",
        "parameters": {},
        "func": "do_sys_info"
    },
    "log_error": {
        "name": "log_error",
        "description": "Log an error message to the error log.",
        "parameters": {
            "msg": "string",
            "trace": "string (optional)"
        },
        "func": "do_log_error"
    },
    "echo": {
        "name": "echo",
        "description": "Echo the input text (for testing).",
        "parameters": {"text": "string"},
        "func": "do_echo"
    },
    "heartbeat": {
        "name": "heartbeat",
        "description": "Get system health info: CPU, memory, disk, uptime.",
        "parameters": {},
        "func": "do_heartbeat"
    },
    "file_read": {
        "name": "file_read",
        "description": "Read a file from the workspace.",
        "parameters": {"path": "string"},
        "func": "do_file_read"
    },
    "file_write": {
        "name": "file_write",
        "description": "Write content to a file (mode: 'w' overwrite, 'a' append).",
        "parameters": {"path": "string", "content": "string", "mode": "string (optional)"},
        "func": "do_file_write"
    },
    "schedule_task": {
        "name": "schedule_task",
        "description": "Schedule a command to run at a specific time or interval. Time format: 'in 5 minutes', 'every day at 10:00', etc. (uses dateparser if installed, otherwise simple timestamps).",
        "parameters": {
            "command": "string",
            "time_spec": "string",
            "job_id": "string (optional)"
        },
        "func": "do_schedule_task"
    },
    "list_scheduled": {
        "name": "list_scheduled",
        "description": "List all scheduled jobs.",
        "parameters": {},
        "func": "do_list_scheduled"
    },
    "remove_scheduled": {
        "name": "remove_scheduled",
        "description": "Remove a scheduled job by its ID.",
        "parameters": {"job_id": "string"},
        "func": "do_remove_scheduled"
    }
}

# ----------------------------------------------------------------------
# Tool implementations

def do_list_tools():
    """Return the list of tools with their metadata."""
    tools_list = []
    for name, info in TOOLS.items():
        tools_list.append({
            "name": name,
            "description": info["description"],
            "parameters": info["parameters"]
        })
    return {"tools": tools_list}

def do_sys_info():
    """Return system information."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "hostname": platform.node()
    }

def do_log_error(msg, trace=""):
    """Append an error to the error log file."""
    entry = {
        "timestamp": time.time(),
        "msg": msg,
        "trace": trace
    }
    try:
        if ERROR_LOG.exists():
            with open(ERROR_LOG) as f:
                log = json.load(f)
        else:
            log = []
        log.append(entry)
        if len(log) > MAX_LOG_ENTRIES:
            log = log[-MAX_LOG_ENTRIES:]
        with open(ERROR_LOG, 'w') as f:
            json.dump(log, f, indent=2)
        return {"status": "logged"}
    except Exception as e:
        return {"error": f"Failed to write log: {e}"}

def do_echo(text):
    """Echo the input."""
    return {"echo": text}

def do_heartbeat():
    """Return system load, memory, disk usage, and uptime."""
    info = {}
    if PSUTIL_AVAILABLE:
        try:
            info["cpu_percent"] = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            info["memory"] = {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent
            }
            disk = psutil.disk_usage('/')
            info["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
            info["uptime_seconds"] = time.time() - psutil.boot_time()
        except Exception as e:
            info["error"] = f"psutil error: {e}"
    else:
        info["error"] = "psutil not installed – install for detailed stats"
        info["platform"] = platform.platform()
    return info

def _safe_path(path):
    """Resolve path relative to SCRIPT_DIR and ensure it stays inside."""
    target = (SCRIPT_DIR / path).resolve()
    try:
        target.relative_to(SCRIPT_DIR)
        return target
    except ValueError:
        return None

def do_file_read(path):
    """Read and return contents of a file (must be inside workspace)."""
    safe = _safe_path(path)
    if not safe:
        return {"error": "Path not allowed (outside workspace)"}
    try:
        with open(safe, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"content": content, "path": str(safe)}
    except Exception as e:
        return {"error": str(e)}

def do_file_write(path, content, mode="w"):
    """Write content to a file (modes: w = overwrite, a = append)."""
    safe = _safe_path(path)
    if not safe:
        return {"error": "Path not allowed (outside workspace)"}
    if mode not in ("w", "a"):
        return {"error": f"Invalid mode '{mode}'; use 'w' or 'a'"}
    try:
        with open(safe, mode, encoding='utf-8') as f:
            f.write(content)
        return {"status": "written", "path": str(safe), "mode": mode}
    except Exception as e:
        return {"error": str(e)}

def do_schedule_task(command, time_spec, job_id=None):
    """
    Add a scheduled job. Simple implementation: store in a JSON file.
    The agent's scheduler will read this file and execute commands when due.
    """
    jobs = []
    if SCHEDULED_JOBS_FILE.exists():
        try:
            with open(SCHEDULED_JOBS_FILE) as f:
                jobs = json.load(f)
        except Exception:
            jobs = []

    if job_id is None:
        job_id = f"job_{int(time.time())}_{len(jobs)}"

    # Parse time_spec – we just store it; the agent's scheduler will interpret.
    # For simplicity, we support:
    #   - "in X minutes/hours/days" -> compute timestamp
    #   - "every day at HH:MM" -> store as cron-like?
    # We'll store raw and let the agent handle it.
    job = {
        "id": job_id,
        "command": command,
        "time_spec": time_spec,
        "created": time.time()
    }
    jobs.append(job)
    try:
        with open(SCHEDULED_JOBS_FILE, 'w') as f:
            json.dump(jobs, f, indent=2)
        return {"status": "scheduled", "job_id": job_id}
    except Exception as e:
        return {"error": f"Failed to write jobs file: {e}"}

def do_list_scheduled():
    """List all scheduled jobs."""
    if not SCHEDULED_JOBS_FILE.exists():
        return {"jobs": []}
    try:
        with open(SCHEDULED_JOBS_FILE) as f:
            jobs = json.load(f)
        return {"jobs": jobs}
    except Exception as e:
        return {"error": f"Failed to read jobs: {e}"}

def do_remove_scheduled(job_id):
    """Remove a scheduled job by ID."""
    if not SCHEDULED_JOBS_FILE.exists():
        return {"error": "No jobs file"}
    try:
        with open(SCHEDULED_JOBS_FILE) as f:
            jobs = json.load(f)
        new_jobs = [j for j in jobs if j.get("id") != job_id]
        if len(new_jobs) == len(jobs):
            return {"error": f"Job ID '{job_id}' not found"}
        with open(SCHEDULED_JOBS_FILE, 'w') as f:
            json.dump(new_jobs, f, indent=2)
        return {"status": "removed", "job_id": job_id}
    except Exception as e:
        return {"error": f"Failed to remove job: {e}"}

# ----------------------------------------------------------------------
# Map function names to actual functions
FUNC_MAP = {
    "do_list_tools": do_list_tools,
    "do_sys_info": do_sys_info,
    "do_log_error": do_log_error,
    "do_echo": do_echo,
    "do_heartbeat": do_heartbeat,
    "do_file_read": do_file_read,
    "do_file_write": do_file_write,
    "do_schedule_task": do_schedule_task,
    "do_list_scheduled": do_list_scheduled,
    "do_remove_scheduled": do_remove_scheduled,
}

# ----------------------------------------------------------------------
# Generic dispatcher using inspect
def main():
    try:
        data = json.loads(sys.stdin.read())
        action = data.get("action")
        if not action:
            print(json.dumps({"error": "No action specified"}))
            return

        tool_info = TOOLS.get(action)
        if not tool_info:
            print(json.dumps({"error": f"Unknown action '{action}'"}))
            return

        func_name = tool_info["func"]
        func = FUNC_MAP.get(func_name)
        if not func:
            print(json.dumps({"error": f"Internal error: unknown function {func_name}"}))
            return

        # Extract parameters expected by the function
        sig = inspect.signature(func)
        kwargs = {}
        for param in sig.parameters.values():
            if param.name in data:
                kwargs[param.name] = data[param.name]
            elif param.default is param.empty:
                # Required parameter missing
                print(json.dumps({"error": f"Missing required parameter '{param.name}'"}))
                return

        result = func(**kwargs)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": f"Multitool exception: {e}"}))

if __name__ == "__main__":
    main()
