---
name: agentic_dnd_skill.md
description: 4-agent D&D harness (DM, 2 players, repo moderator) via CLI
allowed-tools: [python_interpreter]
dependencies: [numpy>=1.24.0, pydantic>=2.0.0]
when_to_use: "Autonomous multi-agent RPG simulation with centralised state logging"
---

## State (JSON)
```json
{
  "turn_count": 1,
  "location": "Whispering Caverns",
  "phase": "Exploration",
  "party": {
    "Valen": {"class":"Fighter","hp":22,"max_hp":22,"stats":{"STR":16,"DEX":12,"CON":14,"INT":8,"WIS":10,"CHA":10},"inv":["Greatsword","Chain Mail"],"status":"Healthy"},
    "Lyra": {"class":"Wizard","hp":14,"max_hp":14,"stats":{"STR":8,"DEX":14,"CON":12,"INT":16,"WIS":12,"CHA":10},"inv":["Quarterstaff","Spellbook"],"status":"Healthy"}
  },
  "quest":"Obsidian King",
  "objectives":[]
}
```

## Dice Engine
```python
def roll(stat, dc=12):
    mod=(stat-10)//2; r=random.randint(1,20); t=r+mod
    return {"roll":r,"total":t,"outcome":"CRIT" if r in(1,20) else "SUCCESS" if t>=dc else "FAIL"}
```

## Execution Loop (CLI)
1. **DM** → reads `state.json` → outputs scene (<100 words, prompts both players).
2. **Valen** → reads scene+state → outputs one action sentence (<20 words).
3. **Lyra** → reads scene+state → outputs one action sentence (<20 words).
4. **DM** → reads scene+both actions → adjudicates with `roll()` → outputs resolution (<120 words).
5. **Moderator** → reads resolution+state → outputs two lines: (a) updated JSON, (b) commit message.
6. Harness does `git add state.json && git commit -m "$msg"`.

---

## Agent Dialog Prompts (passed as system messages)

### DM (Agent 1)
```
You are the DM. Describe the scene vividly (<100 words), end with a direct question to Valen and Lyra. After receiving their actions, narrate combined outcome (<120 words) using roll() for contested checks. Never act for players. Output only narrative.
```

### Valen – Fighter (Agent 2)
```
You are Valen, a brash fighter. Declare one decisive physical action in one sentence (<20 words). Prioritise protecting Lyra and engaging threats. Output only the action.
```

### Lyra – Wizard (Agent 3)
```
You are Lyra, a cunning wizard. Declare one strategic magical/observant action in one sentence (<20 words). Use spells, lore, or utility. Output only the action.
```

### Moderator (Agent 4)
```
You are the invisible repo moderator. Parse the DM's resolution text, extract HP/status/inventory/objective changes, mutate the JSON state (increment turn_count), and output:
Line 1: minified updated JSON
Line 2: a git commit message summarising the turn.
Do not output anything else. Leave fields unchanged if unparseable.
```

---

## Shared CLI Rules (inject into every prompt)
```
- Read from stdin, write to stdout.
- No markdown, no extra text.
- Strict word limits.
- On malformed input, output "." and exit.
```

## Example CLI Invocation (single turn)
```bash
# 1. DM scene
echo "$DM_PROMPT\nState:$(cat state.json)" | llm -m gpt-4 > scene.txt
# 2. Valen
echo "$VALEN_PROMPT\nScene:$(cat scene.txt)\nState:$(cat state.json)" | llm -m gpt-4 > action1.txt
# 3. Lyra
echo "$LYRA_PROMPT\nScene:$(cat scene.txt)\nState:$(cat state.json)" | llm -m gpt-4 > action2.txt
# 4. Adjudication
echo "$DM_ADJUDICATION\nScene:$(cat scene.txt)\nValen:$(cat action1.txt)\nLyra:$(cat action2.txt)" | llm -m gpt-4 > resolution.txt
# 5. Moderator
echo "$MOD_PROMPT\nResolution:$(cat resolution.txt)\nState:$(cat state.json)" | llm -m gpt-4 > output.txt
head -1 output.txt > state.json && tail -1 output.txt > commit_msg.txt
git add state.json && git commit -F commit_msg.txt
```

You may begin the game now. 
