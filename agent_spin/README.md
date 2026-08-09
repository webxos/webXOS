# 🌀 AGENT SPIN – Autonomous Code Factory (Under Development and Testing)
**Infinite Loop • No Timeouts • Git Auto‑Commit • Local LLM Driven**

AGENT SPIN is a **self‑running, continuously building code generator** that uses a local Ollama model to autonomously write, test, fix, and version‑control complete software projects. Just set your preferred model once, give it a goal, and watch it create a new project every cycle – until closed.

---

## ✨ Features

- 🧠 **Uses any Ollama model** – pick your favorite, set it in the script, and go.
- ♾️ **Infinite loop** – keeps generating new iterations until you press `Ctrl+C`.
- ⏳ **No timeouts** – waits indefinitely for even the slowest models to respond.
- 📁 **Each cycle = a fresh project** – all source code and tests in a new folder.
- 🛠️ **Automatic test‑and‑repair** – if tests fail, the LLM patches the code up to 3 times.
- 📦 **Git version control out of the box** – every successful cycle is committed.
- 🚀 **Optional auto‑push** – un-comment one line to push commits to a remote repo.
- 🧹 **Zero extra commands** – no model picker, no CLI options; edit the script once and run.

---

## 📦 Requirements

- [Ollama](https://ollama.com/) installed and running, with at least one model downloaded (e.g., `llama3.2`, `qwen2.5`, etc.).
- `curl` (for API communication).
- `git` (optional – skip if not needed).
- Bash 3+ (Linux, macOS, WSL).

---

## 🚀 Quick Start

1. **Download** or copy the `agent_spin.sh` script.

2. **Edit the model** (optional – default is `qwen2.5:0.5b`):
   ```bash
   MODEL_NAME="llama3.2"      # change to your preferred model
   ```

3. **Run it** (no `chmod` needed):
   ```bash
   bash agent_spin.sh
   ```

4. **Answer the goal question** – e.g., *“a REST API in Python”*.

5. **Let it run** – the script will cycle indefinitely, creating new projects and committing successes.

6. **Stop** – press `Ctrl+C` at any time.

---

## 🔧 Configuration (directly inside the script)

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | API endpoint for Ollama | `http://localhost:11434/api/generate` |
| `MODEL_NAME` | Model to use | `qwen2.5:0.5b` |
| `MAX_TEST_RETRIES` | Repair attempts per cycle | `3` |

### Enabling Auto‑Push to Remote

By default, commits stay local. To push each successful commit to a remote repository:

1. Un-comment this line inside `setup_workspace()` (around line 70):
   ```bash
   # git remote add origin https://github.com/your/repo.git
   ```
2. Replace the URL with your own.
3. The script will automatically `git push` after every successful cycle.

---

## 📂 Output Structure

A new workspace is created on your Desktop:

```
~/Desktop/AGENT_FORGE_<hash>_<timestamp>/
├── agent.log                # Full activity log
├── README.md                # Workspace readme
├── Build_1/                 # First cycle
│   ├── app.py / app.js / app.sh
│   └── test_app.py / test_app.js / test_app.sh
├── Build_2/
│   └── ...
└── .git/                    # Git repository (if Git installed)
```

Each `Build_N` is a self‑contained project. The script never overwrites previous cycles – it only adds new ones.

---

## ⚙️ Automation Pipeline

1. **Prompt** – The LLM chooses a runtime (`python`, `nodejs`, or `bash`) based on your goal.
2. **Build** – The LLM generates the application and a corresponding test file.
3. **Test** – The tests are executed. If they fail, the LLM is asked to fix the code (up to 3 attempts).
4. **Commit** – If tests pass, the entire workspace is `git add`ed and committed with a descriptive message (e.g., `Cycle 5: Build_5 (python)`). If a remote is configured, it is pushed.
5. **Loop** – The cycle repeats indefinitely.

---

## ⏳ No‑Timeout Design

The script uses `curl --max-time 0` inside `ask_ollama()`, which means it **waits forever** for the model to finish generating.  
This is ideal for slow or very large models that can take several minutes per response – AGENT SPIN will patiently wait.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **Error: `Cannot reach Ollama`** | Ensure Ollama is running (`ollama serve`). |
| **Model not found** | Verify the model name in the script exists locally (`ollama list`). |
| **Git commit fails** | The script warns but continues – version control is optional. |
| **No output / hanging** | The model may be slow – check the log file (`agent.log`) inside the workspace. |
| **Permission errors** | Run with `bash` (no `chmod` needed). For Bash projects, the script already sets executable bits. |

---

*Disclaimer: Vibe coded with Gemini + Deepseek*

## 📄 License

MIT
