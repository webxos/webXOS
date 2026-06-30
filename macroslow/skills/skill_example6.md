**Here is a full complete `skill.md`**

```markdown
# MAML Skill: Personal Productivity Assistant

**Skill Name:** `personal-productivity-assistant`  
**Version:** 1.0.0  
**Type:** `skill`  
**Target Harnesses:** Local Harness or Agent
**Purpose:** Summarize meetings, emails, or notes and extract actionable items with user-specific context.

---

## Intent

Provide lightweight, personalized summarization and action item extraction for productivity tasks such as meetings, emails, or daily notes.

---

## Context

Designed for personal agent use. Leverages user context (preferences, priorities) stored in Hermes memory or passed in the payload. Keeps computation minimal for fast response on local hardware.

---

## Code_Blocks

```python
from typing import Dict, Any, List

def summarize_meeting(transcript: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize transcript and extract action items with personalization.
    """
    # Simple keyword-based extraction (expand with LLM or NLP as needed)
    key_points = extract_key_points(transcript)
    action_items = extract_actions(transcript)
    
    # Apply user preferences
    personalized_summary = apply_user_preferences(key_points, user_context)
    prioritized_actions = prioritize_actions(action_items, user_context.get("priorities", []))
    
    return {
        "status": "success",
        "summary": personalized_summary,
        "action_items": prioritized_actions,
        "key_points_count": len(key_points),
        "action_items_count": len(prioritized_actions)
    }

def extract_key_points(transcript: str) -> List[str]:
    """Basic key point extraction"""
    # Placeholder - replace with real NLP or LLM call
    sentences = transcript.split('. ')
    return [s.strip() for s in sentences[:5] if len(s) > 10]

def extract_actions(transcript: str) -> List[Dict[str, str]]:
    """Extract action items"""
    # Placeholder logic
    return [
        {"action": "Follow up with team", "owner": "user", "deadline": "this week"},
        {"action": "Review document", "owner": "user", "deadline": "tomorrow"}
    ]

def apply_user_preferences(key_points: List[str], user_context: Dict) -> str:
    """Personalize summary based on user preferences"""
    focus_areas = user_context.get("focus_areas", ["productivity", "decisions"])
    summary = "Meeting Summary:\n" + "\n".join(key_points[:3])
    return summary

def prioritize_actions(actions: List[Dict], priorities: List[str]) -> List[Dict]:
    """Prioritize actions based on user priorities"""
    return sorted(actions, key=lambda x: x.get("deadline", ""))  # simple sort
```

---

## Input_Schema

```json
{
  "type": "object",
  "properties": {
    "transcript": { "type": "string", "description": "Meeting transcript or email content" },
    "user_context": { 
      "type": "object", 
      "description": "User preferences, priorities, and history" 
    }
  },
  "required": ["transcript"]
}
```

## Output_Schema

```json
{
  "type": "object",
  "properties": {
    "status": { "type": "string" },
    "summary": { "type": "string" },
    "action_items": { "type": "array" },
    "key_points_count": { "type": "integer" },
    "action_items_count": { "type": "integer" }
  }
}
```

---

## History

- 2026-06-29
