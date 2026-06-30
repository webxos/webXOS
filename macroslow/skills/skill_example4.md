**Here is the full complete `skill.md`**

```markdown
# MAML Skill: Adaptive Memory Skill

**Skill Name:** `adaptive-memory-skill`  
**Version:** 1.0.0  
**Type:** `skill`  
**Target Harnesses:** Hermes (primary - memory & reflection), OpenClaw (secondary)  
**Purpose:** Improve performance over time by reading past execution history and adapting parameters dynamically.

---

## Intent

Create a self-improving skill that analyzes previous runs stored in the MAML History section and adjusts its behavior for better future results.

---

## Context

Optimized for Hermes agents that maintain long-term memory. The skill reads historical performance data and adapts thresholds, strategies, or parameters accordingly. This enables continuous improvement without external retraining.

---

## Code_Blocks

```python
from typing import Dict, Any, List
import json

def extract_success_rate(history: List[Dict]) -> float:
    """Extract average success rate from past History entries"""
    if not history:
        return 0.5  # default neutral value
    successes = sum(1 for entry in history if "success" in entry.get("message", "").lower())
    return successes / len(history) if history else 0.5

def adaptive_process(data: Any, history: List[Dict]) -> Dict[str, Any]:
    """
    Adaptive processing based on past performance.
    Reads History to adjust behavior.
    """
    previous_success = extract_success_rate(history)
    
    # Dynamic threshold adaptation
    if previous_success > 0.9:
        threshold = 0.85
        strategy = "aggressive"
    elif previous_success > 0.7:
        threshold = 0.75
        strategy = "balanced"
    else:
        threshold = 0.65
        strategy = "conservative"
    
    # Apply adapted logic
    result = process_with_adapted_threshold(data, threshold)
    
    return {
        "status": "success",
        "previous_success_rate": round(previous_success, 4),
        "adapted_threshold": threshold,
        "strategy": strategy,
        "result": result,
        "adaptation_note": f"Adapted using {len(history)} past runs"
    }

def process_with_adapted_threshold(data: Any, threshold: float):
    """Core processing logic with dynamic parameter"""
    # Placeholder for real task logic (e.g., filtering, scoring, etc.)
    score = calculate_base_score(data)
    return {
        "final_score": score,
        "passed_adaptation": score >= threshold,
        "threshold_used": threshold
    }

def calculate_base_score(data: Any):
    """Example scoring function - replace with real implementation"""
    return 0.82  # placeholder
```

---

## Input_Schema

```json
{
  "type": "object",
  "properties": {
    "data": { "type": "object", "description": "Input data for processing" },
    "history": { 
      "type": "array", 
      "description": "Previous History entries from this MAML file" 
    }
  },
  "required": ["data"]
}
```

## Output_Schema

```json
{
  "type": "object",
  "properties": {
    "status": { "type": "string" },
    "previous_success_rate": { "type": "number" },
    "adapted_threshold": { "type": "number" },
    "strategy": { "type": "string" },
    "result": { "type": "object" },
    "adaptation_note": { "type": "string" }
  }
}
```

---

## History

- 2026-06-29
