# 🌀 agent_spin.sh – Autonomous Code Factory

**AGENT SPIN** is a self‑improving, infinite‑loop code generator that uses **Ollama** (local LLMs) to autonomously build, test, fix, and version‑control entire software projects—**completely hands‑off**.

Simply tell it **what to build**, and it will continuously create new iterations, each in its own folder, with source code and unit tests. Every successful build is automatically **committed to Git** and (optionally) **pushed to a remote repository** in real time.

---

## ✨ Features

- ✅ **Zero‑interaction** after the initial goal – runs forever until you press Ctrl+C.
- 🧠 **Uses any Ollama model** – just set the model name once in the script.
- 🔁 **Infinite loop**: each cycle creates a new project folder with:
  - Main application (`app.py`, `app.js`, or `app.sh`)
  - A test file (`test_app.py`, `test_app.js`, or `test_app.sh`)
- 🛠️ **Automatic test‑and‑fix** – if tests fail, the script asks the LLM to patch the code up to 3 times.
- 📦 **Git version control** – initializes a local Git repository in the workspace and commits each successful cycle.
- 🚀 **Optional auto‑push** – uncomment one line to push every commit to a remote (GitHub, GitLab, etc.).
- 📁 **Unique workspace** – each run creates a new timestamped folder on your Desktop.

---

## 📦 Requirements

- [Ollama](https://ollama.com/) running locally with at least one model downloaded (e.g., `qwen2.5:0.5b`, `llama3.2`, etc.).
- `curl` – to talk to the Ollama API.
- `git` (optional) – for version control; if not installed, the script skips Git.
- Bash 3+ (Linux/macOS/WSL).

---

## 🚀 Quick Start

1. **Download or copy** the [`agent_spin.sh`](#) script.

2. **Edit the model** (optional):  
   Open the script and change the `MODEL_NAME` variable (line ~8) to your preferred Ollama model:
   ```bash
   MODEL_NAME="qwen2.5:0.5b"
   ```

3. **Run it** (no `chmod` needed):
   ```bash
   bash agent_spin.sh
   ```

4. **Answer the prompt** – tell AGENT SPIN what to build (e.g., *“a JSON parser in Python”*).

5. **Watch it go** – the script will cycle forever, creating new projects and committing successes.

6. **Stop** – press `Ctrl+C` at any time.

---

## 🔧 Configuration (inside the script)

| Variable | Description |
|----------|-------------|
| `OLLAMA_URL` | Ollama API endpoint (default: `http://localhost:11434/api/generate`). |
| `MODEL_NAME` | The Ollama model to use (edit this to your preferred one). |
| `MAX_TEST_RETRIES` | Number of repair attempts per cycle (default: 3). |

### Enabling Auto‑Push to Remote

By default, commits stay local. To push each successful cycle to a remote repo:

1. Uncomment this line inside `setup_workspace()` (around line 70):
   ```bash
   # git remote add origin https://github.com/your/repo.git
   ```
2. Replace the URL with your own (e.g., your GitHub repo).
3. The script will automatically `git push origin main` (or `master`) after each commit.

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
└── .git/                    # Git repository (if Git is installed)
```

Each `Build_N` folder is a self‑contained project. The script never modifies previous cycles – it only adds new ones.

---

## ⚙️ How It Works (Cycle)

1. **Ideate**: The LLM decides which runtime to use (`python`, `nodejs`, or `bash`) based on your original goal.
2. **Build**: The LLM generates an application file and a corresponding test file.
3. **Test**: The tests are executed. If they fail, the script asks the LLM to fix the code and retries (up to 3 times).
4. **Commit**: If tests pass, the entire workspace is `git add`ed and committed with a descriptive message (e.g., `Cycle 5: Build_5 (python)`). If a remote is configured, it is pushed.
5. **Loop**: The cycle repeats indefinitely.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **Error: `Cannot reach Ollama`** | Make sure Ollama is running (`ollama serve` or `ollama run <model>`). |
| **Model not found** | Ensure the model name in the script exists locally (`ollama list`). |
| **Git commit fails** | The script will warn but continue – no action needed unless you want version control. |
| **No output / hanging** | The script uses `curl` with a timeout – if the model is slow, be patient. You can also check the log file inside the workspace. |
| **Permission errors** | Run with `bash` (no `chmod` needed). If you get `./app.sh: Permission denied`, the script already runs `chmod +x` for Bash projects. |

---

*Disclaimer: This app was vibe coded with Gemini + Deepseek*

## 📄 License

MIT
