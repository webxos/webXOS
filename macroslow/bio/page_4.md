# MAML-Enhanced bio.md: Building the Perfect Living Agent Memory System with Markdown as Medium Language (.maml.md)

**Universal, Offline-First Guide for Persistent, Self-Evolving Agentic Intelligence**  
**Prepared for Local Ollama Harnesses, Skill Developers, and Bioinformatics-Inspired AI Builders**  
**Focus: Transforming Traditional bio.md into Executable, DP-Optimized .maml.md Workflows**  
**Compatibility: Hermes Memory Curation, OpenClaw Gateways, Pure Local Ollama (No Internet)**  
**Version: 2.1 (Advanced MAML + Dynamic Programming Integration, July 2026)**  
**Page 4 of 10**

## 4. Harness Integration Patterns with Python/Bash + Ollama for bio.maml.md

This page details practical integration of the MAML-enhanced `bio.maml.md` into a complete agent harness. We provide ready-to-use code patterns for Python (recommended primary harness) and Bash helpers, covering bootstrap, heartbeat lifecycle, Ollama injection, DP engine calls, and advanced features like real-time updates via inotify.

The harness acts as the "body" while bio.maml.md is the "mind" — parsing, validating, executing, and updating the living file on every turn.

### Core Harness Architecture (Python Example)

```python
# harness.py - Main agent loop (expandable to Rust/JS)
import yaml
import json
import subprocess
import ollama
from datetime import datetime
import os
from pathlib import Path

BIO_FILE = "bio.maml.md"
DATA_DIR = Path("data")

def load_bio_maml() -> dict:
    """Parse front matter + full content."""
    with open(BIO_FILE, "r") as f:
        content = f.read()
    # Split front matter
    if content.startswith('---'):
        parts = content.split('---', 2)
        front_matter = yaml.safe_load(parts[1])
        body = parts[2].strip() if len(parts) > 2 else ""
    else:
        front_matter = {}
        body = content
    return {"front_matter": front_matter, "body": body, "raw": content}

def validate_and_inject_system_prompt(bio: dict) -> str:
    """Extract relevant sections for Ollama system prompt."""
    # Prioritize SOUL + SKILLS + recent MEMORY + CONTEXT
    # For efficiency: summarize long sections if needed
    prompt = bio["body"]  # Or targeted extraction
    # Apply pruning if context too large
    if len(prompt) > 8000:  # tokens rough estimate
        prompt = bio["front_matter"].get("pruning_strategy", "truncate") 
    return prompt

def run_heartbeat(user_input: str) -> str:
    """Full lifecycle for one interaction."""
    bio = load_bio_maml()
    
    # Build messages with bio as system
    system_prompt = validate_and_inject_system_prompt(bio)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # Call local Ollama
    response = ollama.chat(model='qwen2.5:1.5b', messages=messages)  # or larger model
    assistant_reply = response['message']['content']
    
    # Process heartbeat
    heartbeat_data = {
        "user_input": user_input,
        "response": assistant_reply,
        "timestamp": datetime.now().isoformat()
    }
    
    # Execute relevant Code_Blocks from bio (or dedicated skill)
    # Example: call DP processor
    result = execute_code_block("process_heartbeat", heartbeat_data)  # Harness helper
    
    # Update bio.maml.md incrementally
    update_bio_maml(result, assistant_reply)
    
    # Append to History
    append_to_history(heartbeat_data, result)
    
    return assistant_reply

def execute_code_block(function_name: str, input_data: dict) -> dict:
    """Sandbox execution of Code_Blocks from SKILLS."""
    # In production: parse bio.maml.md for matching block, exec in subprocess
    # Simplified:
    try:
        # Call Python DP/heartbeat function
        proc = subprocess.run(["python3", "-c", f"""
from bio_skills import process_heartbeat
import json
print(json.dumps(process_heartbeat({json.dumps(input_data)})))
        """], capture_output=True, text=True, timeout=60)
        return json.loads(proc.stdout)
    except Exception as e:
        return {"status": "error", "message": str(e)}

def update_bio_maml(result: dict, reply: str):
    """Incremental, safe update using DP insights."""
    bio = load_bio_maml()
    # Update timestamp, DP metadata, append summary to MEMORY
    # Use sed-like or file editing logic
    with open(BIO_FILE, "a") as f:  # Better: targeted replace
        pass  # Implement section-aware editor
    # Update front_matter dp_metadata with new alignment score

def append_to_history(data: dict, result: dict):
    """Immutable History append."""
    entry = f"- {data['timestamp']}: [HEARTBEAT] User query processed. DP Score: {result.get('dp_alignment', {}).get('alignment_score', 0):.2f}"
    # Append under ## History section (parse and insert)
    pass

# Bootstrap function
def bootstrap():
    if not os.path.exists(BIO_FILE):
        # Create rich template from original bio.md guide + MAML
        print("Creating initial bio.maml.md...")
        # Write full template with SOUL, SKILLS, etc.
        # ...

if __name__ == "__main__":
    bootstrap()
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        reply = run_heartbeat(user_input)
        print("Agent:", reply)
```

### Bash Helpers for Lightweight Harnesses & Cron

```bash
# memory_sync.sh - Hourly consolidation
#!/bin/bash
BIO="bio.maml.md"
LOG="data/logs/heartbeats.raw.log"

# Summarize recent logs with Ollama
tail -n 50 "$LOG" > /tmp/recent.txt
SUMMARY=$(ollama run qwen2.5:0.5b "Summarize these heartbeats for long-term MEMORY, use DP-friendly concise format: $(cat /tmp/recent.txt)")

# Append to ## MEMORY section (incremental)
sed -i "/## MEMORY/a\\- $SUMMARY" "$BIO"

# Prune old logs
> "$LOG"  # Or archive
echo "Memory consolidated at $(date)"
```

### Advanced Integration Features
- **Inotify Real-Time Context Updates**: Watch key files (e.g., git repo) and trigger CONTEXT section refresh.
- **Session Branching**: On user request, fork a new bio.maml.md copy with updated SESSION TREE.
- **Hermes/OpenClaw Compatibility**: Export bio.maml.md as MCP payload or import skills from external .maml.md files.
- **Error Recovery**: If update fails, rollback using History + git (optional versioning of bio file itself).
- **Performance Tuning**: Cache parsed sections; run heavy DP in background; use small models for routine tasks.

### Full Heartbeat Lifecycle Summary
1. Load & validate bio.maml.md.
2. Construct system prompt from sections.
3. Ollama inference.
4. Execute Code_Blocks (DP alignment + summarization).
5. Incremental update to file + History.
6. Optional cron consolidation.

This architecture makes the harness lightweight yet extremely powerful. The bio.maml.md drives intelligence while the harness provides structure and execution safety.

**(End of Page 4. Continued on Page 5: Advanced Dynamic Programming Algorithms for Memory Alignment, Skill Binding, and Context Folding in bio.maml.md.)**
