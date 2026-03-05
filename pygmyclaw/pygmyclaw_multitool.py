#!/usr/bin/env python3
"""
PygmyClaw Multitool – Contains the actual tool implementations.
"""
import json
import sys
import os
import platform
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ERROR_LOG = SCRIPT_DIR / "error_log.json"
MAX_LOG_ENTRIES = 1000

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
    }
}

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
        # Keep only last MAX_LOG_ENTRIES
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

# ----------------------------------------------------------------------
# Main dispatcher
def main():
    try:
        data = json.loads(sys.stdin.read())
        action = data.get("action")
        if not action:
            print(json.dumps({"error": "No action specified"}))
            return

        # Find the tool
        tool_info = TOOLS.get(action)
        if not tool_info:
            print(json.dumps({"error": f"Unknown action '{action}'"}))
            return

        # Call the corresponding function
        func_name = tool_info["func"]
        if func_name == "do_list_tools":
            result = do_list_tools()
        elif func_name == "do_sys_info":
            result = do_sys_info()
        elif func_name == "do_log_error":
            msg = data.get("msg")
            trace = data.get("trace", "")
            if msg is None:
                result = {"error": "Missing 'msg' parameter"}
            else:
                result = do_log_error(msg, trace)
        elif func_name == "do_echo":
            text = data.get("text")
            if text is None:
                result = {"error": "Missing 'text' parameter"}
            else:
                result = do_echo(text)
        else:
            result = {"error": f"Internal error: unknown function {func_name}"}

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": f"Multitool exception: {e}"}))

if __name__ == "__main__":
    main()
