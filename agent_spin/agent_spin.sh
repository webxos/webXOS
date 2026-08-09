#!/bin/bash
# ==============================================================================
# AGENT SPIN – Infinite Loop, Single Project Folder, No Timeouts
# ==============================================================================

# --- CONFIGURATION ------------------------------------------------------------
OLLAMA_URL="http://localhost:11434/api/generate"
MODEL_NAME="qwen2.5:0.5b"          # <-- Change to your preferred model
MAX_TEST_RETRIES=3
CYCLE_COUNT=0
USER_GOAL=""
WORKSPACE_DIR=""
LOG_FILE=""
PROJECT_DIR=""                     # single folder for all cycles

# --- Cleanup on exit ----------------------------------------------------------
cleanup() {
    echo -e "\n🛑 AGENT SPIN stopped after $CYCLE_COUNT cycles."
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- Check Ollama -------------------------------------------------------------
check_ollama() {
    if ! curl -s --connect-timeout 2 "$OLLAMA_URL" > /dev/null; then
        echo "❌ ERROR: Cannot reach Ollama at $OLLAMA_URL"
        echo "   Make sure Ollama is running (e.g., 'ollama serve')"
        exit 1
    fi
}

# --- Get user goal ------------------------------------------------------------
get_goal() {
    echo "======================================================================"
    echo "🌀 AGENT SPIN (Model: $MODEL_NAME)"
    echo "======================================================================"
    echo "💡 What should I build for you?"
    echo -n "> "
    read -r USER_GOAL
    if [ -z "$USER_GOAL" ]; then
        USER_GOAL="Automated file processing framework with comprehensive tests"
        echo "   Using default goal: $USER_GOAL"
    fi
}

# --- Setup workspace & Git repo -----------------------------------------------
setup_workspace() {
    GEN_HASH=$((RANDOM % 900000 + 100000))
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    WORKSPACE_DIR="$HOME/Desktop/AGENT_FORGE_${GEN_HASH}_${TIMESTAMP}"
    PROJECT_DIR="$WORKSPACE_DIR/project"   # single folder for all cycles
    LOG_FILE="$WORKSPACE_DIR/agent.log"
    mkdir -p "$PROJECT_DIR"
    touch "$LOG_FILE"
    echo "[$(date)] AGENT SPIN started with model: $MODEL_NAME" >> "$LOG_FILE"
    echo "📁 Workspace: $WORKSPACE_DIR"
    echo "📂 Project folder: $PROJECT_DIR (overwritten each cycle)"

    # Initialize Git repo inside the project folder
    if command -v git >/dev/null; then
        echo "🔧 Initializing Git repository..."
        cd "$PROJECT_DIR" || return
        git init -q
        git config user.email "agent@spin.local"
        git config user.name "AGENT SPIN"
        echo "# AGENT SPIN Workspace" > README.md
        echo "Goal: $USER_GOAL" >> README.md
        git add README.md
        git commit -q -m "Initial commit"
        echo "✅ Git repository ready"
        # Optionally add a remote for auto‑push:
        # git remote add origin https://github.com/your/repo.git
        cd - > /dev/null || return
    else
        echo "⚠️  Git not found – skipping version control."
    fi
}

# --- Git commit after each successful cycle -----------------------------------
git_commit() {
    if ! command -v git >/dev/null; then
        return
    fi
    cd "$PROJECT_DIR" || return
    git add .
    local msg="Cycle $CYCLE_COUNT: $PROJ_LANG - $USER_GOAL"
    if git commit -q -m "$msg"; then
        echo "📦 Committed: $msg"
        # Push if remote configured
        if git remote | grep -q origin; then
            if git push -q origin main 2>/dev/null || git push -q origin master 2>/dev/null; then
                echo "🚀 Pushed to remote"
            else
                echo "⚠️  Push failed"
            fi
        fi
    else
        echo "⚠️  Nothing new to commit (no changes?)"
    fi
    cd - > /dev/null || return
}

# --- Query Ollama – INFINITE TIMEOUT ------------------------------------------
ask_ollama() {
    local sys_prompt="$1"
    local user_prompt="$2"
    local full="${sys_prompt}\n\nUser: ${user_prompt}"
    local escaped
    escaped=$(echo "$full" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    local response
    response=$(curl -s --max-time 0 --connect-timeout 60 -X POST "$OLLAMA_URL" \
        -d "{\"model\": \"$MODEL_NAME\", \"prompt\": \"$escaped\", \"stream\": false}" \
        2>/dev/null)
    if [ -z "$response" ]; then
        echo "⚠️  Ollama request failed." >&2
        return 1
    fi
    echo "$response" | grep -o '"response":"[^"]*"' | sed 's/"response":"//;s/"$//' \
        | sed 's/\\n/\n/g;s/\\t/\t/g;s/\\"/"/g;s/\\\\/\\/g'
}

# --- Main loop phases (everything runs inside PROJECT_DIR) --------------------
ideate() {
    ((CYCLE_COUNT++))
    echo -e "\n----------------------------------------------------------------------"
    echo "🔄 CYCLE #$CYCLE_COUNT – Starting fresh..."
    echo "----------------------------------------------------------------------"

    local lang_choice
    lang_choice=$(ask_ollama \
        "You are a software architect. Output only one word: python, nodejs, or bash." \
        "For goal: '$USER_GOAL', which runtime should I use?")
    if [ $? -ne 0 ] || [ -z "$lang_choice" ]; then
        PROJ_LANG="${PROJ_LANG:-bash}"
    else
        PROJ_LANG=$(echo "$lang_choice" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
        case "$PROJ_LANG" in
            python|nodejs|bash) ;;
            *) PROJ_LANG="bash" ;;
        esac
    fi

    # Clear the project folder for a completely fresh start
    cd "$PROJECT_DIR" || return 1
    rm -rf ./* 2>/dev/null   # wipe everything except .git
    # Keep .git folder if it exists
    if [ -d "../.git" ]; then
        mv ../.git ./ 2>/dev/null
    fi
    echo "📦 Reset project folder. Runtime: $PROJ_LANG"
    cd - > /dev/null || return 1
}

build() {
    echo "🤖 Generating code (this may take a while) ..."
    local app_file test_file
    case "$PROJ_LANG" in
        python)
            app_file="app.py"
            test_file="test_app.py"
            TEST_CMD="python3 test_app.py"
            RUN_CMD="python3 app.py"
            ;;
        nodejs)
            app_file="app.js"
            test_file="test_app.js"
            TEST_CMD="node test_app.js"
            RUN_CMD="node app.js"
            ;;
        *)
            app_file="app.sh"
            test_file="test_app.sh"
            TEST_CMD="./test_app.sh"
            RUN_CMD="./app.sh"
            ;;
    esac

    cd "$PROJECT_DIR" || return 1
    local sys="Output only raw source code. No markdown, no explanations."
    ask_ollama "$sys" "Write a $PROJ_LANG script for: $USER_GOAL" > "$app_file"
    ask_ollama "$sys" "Write a test script named $test_file that validates $app_file" > "$test_file"

    if [ "$PROJ_LANG" = "bash" ]; then
        chmod +x "$app_file" "$test_file"
    fi
    cd - > /dev/null || return 1
}

test_and_fix() {
    echo "🧪 Testing..."
    cd "$PROJECT_DIR" || return 1
    local attempts=0
    local status=1
    local target_file
    target_file=$(echo "$RUN_CMD" | awk '{print $NF}')

    while [ $attempts -lt $MAX_TEST_RETRIES ] && [ $status -ne 0 ]; do
        ((attempts++))
        echo "🔍 [Attempt $attempts/$MAX_TEST_RETRIES] Running: $TEST_CMD"
        eval "$TEST_CMD" > test_output.log 2>&1
        status=$?
        if [ $status -eq 0 ]; then
            echo "✅ SUCCESS"
            echo "[$(date)] Cycle $CYCLE_COUNT passed" >> "$LOG_FILE"
            break
        elif [ $attempts -lt $MAX_TEST_RETRIES ]; then
            echo "🔧 Fixing (may take time) ..."
            local errors code
            errors=$(cat test_output.log 2>/dev/null)
            code=$(cat "$target_file" 2>/dev/null)
            ask_ollama \
                "You are a debugger. Output only fixed code. No markdown." \
                "Fix this $PROJ_LANG code:\n$code\n\nErrors:\n$errors" > "$target_file"
            [ "$PROJ_LANG" = "bash" ] && chmod +x "$target_file"
        fi
    done

    if [ $status -ne 0 ]; then
        echo "❌ FAILED after $MAX_TEST_RETRIES attempts"
        echo "[$(date)] Cycle $CYCLE_COUNT failed" >> "$LOG_FILE"
    fi
    cd - > /dev/null || return 1
    return $status
}

cleanup_cycle() {
    rm -f "$PROJECT_DIR/test_output.log"
    sleep 0.5
    echo "✅ Cycle $CYCLE_COUNT complete."
}

# --- Main ---------------------------------------------------------------------
main() {
    check_ollama
    get_goal
    setup_workspace

    echo "======================================================================"
    echo "🚀 Running forever. Press Ctrl+C to stop."
    echo "   Each cycle will overwrite the project folder and commit if tests pass."
    echo "   ⏳ The script will wait indefinitely for the model to respond."
    echo "======================================================================"

    while true; do
        ideate
        build
        if test_and_fix; then
            git_commit
        else
            echo "⚠️  Cycle $CYCLE_COUNT failed – skipping commit."
        fi
        cleanup_cycle
    done
}

main
