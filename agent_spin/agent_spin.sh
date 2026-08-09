#!/bin/bash
# ==============================================================================
# AGENT SPIN – Autonomous Code Factory with Git auto‑commit & push
# ==============================================================================

# --- CONFIGURATION ------------------------------------------------------------
OLLAMA_URL="http://localhost:11434/api/generate"
MODEL_NAME="qwen2.5:0.5b"          # <-- Change this to your preferred model
MAX_TEST_RETRIES=3
CYCLE_COUNT=0
USER_GOAL=""
WORKSPACE_DIR=""
LOG_FILE=""

# --- Cleanup on exit ----------------------------------------------------------
cleanup() {
    echo -e "\n🛑 AGENT SPIN stopped."
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
    LOG_FILE="$WORKSPACE_DIR/agent.log"
    mkdir -p "$WORKSPACE_DIR"
    touch "$LOG_FILE"
    echo "[$(date)] AGENT SPIN started with model: $MODEL_NAME" >> "$LOG_FILE"
    echo "📁 Workspace: $WORKSPACE_DIR"

    # Initialize Git repo if Git is available
    if command -v git >/dev/null; then
        echo "🔧 Initializing Git repository..."
        cd "$WORKSPACE_DIR" || return
        git init -q
        git config user.email "agent@spin.local"
        git config user.name "AGENT SPIN"
        echo "# AGENT SPIN Workspace" > README.md
        echo "Goal: $USER_GOAL" >> README.md
        git add README.md
        git commit -q -m "Initial commit"
        echo "✅ Git repository ready"
        # Optionally add a remote if you want auto‑push:
        # git remote add origin https://github.com/your/repo.git
        # (uncomment and set your URL above)
        cd - > /dev/null || return
    else
        echo "⚠️  Git not found – skipping version control."
    fi
}

# --- Git commit & push after each cycle ---------------------------------------
git_commit() {
    if ! command -v git >/dev/null; then
        return
    fi
    cd "$WORKSPACE_DIR" || return
    # Add all changes (new project folders, updated files)
    git add .
    # Commit with a descriptive message
    local msg="Cycle $CYCLE_COUNT: $PROJ_NAME ($PROJ_LANG)"
    if git commit -q -m "$msg"; then
        echo "📦 Committed: $msg"
        # Push if a remote is configured
        if git remote | grep -q origin; then
            if git push -q origin main 2>/dev/null || git push -q origin master 2>/dev/null; then
                echo "🚀 Pushed to remote"
            else
                echo "⚠️  Push failed (no remote or network issue)"
            fi
        fi
    else
        echo "⚠️  Nothing new to commit."
    fi
    cd - > /dev/null || return
}

# --- Query Ollama -------------------------------------------------------------
ask_ollama() {
    local sys_prompt="$1"
    local user_prompt="$2"
    local full="${sys_prompt}\n\nUser: ${user_prompt}"
    local escaped
    escaped=$(echo "$full" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    local response
    response=$(curl -s -X POST "$OLLAMA_URL" \
        -d "{\"model\": \"$MODEL_NAME\", \"prompt\": \"$escaped\", \"stream\": false}" \
        2>/dev/null)
    if [ -z "$response" ]; then
        echo "⚠️  Ollama request failed." >&2
        return 1
    fi
    echo "$response" | grep -o '"response":"[^"]*"' | sed 's/"response":"//;s/"$//' \
        | sed 's/\\n/\n/g;s/\\t/\t/g;s/\\"/"/g;s/\\\\/\\/g'
}

# --- Main loop phases ---------------------------------------------------------
ideate() {
    ((CYCLE_COUNT++))
    echo -e "\n----------------------------------------------------------------------"
    echo "🔄 CYCLE #$CYCLE_COUNT"
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

    PROJ_NAME="Build_${CYCLE_COUNT}"
    PROJ_DIR="$WORKSPACE_DIR/$PROJ_NAME"
    mkdir -p "$PROJ_DIR"
    echo "📦 Project: $PROJ_NAME  |  Runtime: $PROJ_LANG"
}

build() {
    echo "🤖 Generating code..."
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

    local sys="Output only raw source code. No markdown, no explanations."
    ask_ollama "$sys" "Write a $PROJ_LANG script for: $USER_GOAL" > "$PROJ_DIR/$app_file"
    ask_ollama "$sys" "Write a test script named $test_file that validates $app_file" > "$PROJ_DIR/$test_file"

    if [ "$PROJ_LANG" = "bash" ]; then
        chmod +x "$PROJ_DIR/$app_file" "$PROJ_DIR/$test_file"
    fi
}

test_and_fix() {
    echo "🧪 Testing..."
    cd "$PROJ_DIR" || return 1
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
            echo "🔧 Fixing..."
            local errors logs code
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
    return $status   # return test status so we can decide to commit only on success
}

cleanup_cycle() {
    rm -f "$PROJ_DIR/test_output.log"
    sleep 0.5
}

# --- Main ---------------------------------------------------------------------
main() {
    check_ollama
    get_goal
    setup_workspace

    echo "======================================================================"
    echo "🚀 Running forever. Press Ctrl+C to stop."
    echo "   Each successful cycle will be committed to Git (and pushed if remote set)."
    echo "======================================================================"

    while true; do
        ideate
        build
        if test_and_fix; then
            # Only commit if tests passed
            git_commit
        else
            echo "⚠️  Cycle $CYCLE_COUNT failed – skipping commit."
        fi
        cleanup_cycle
    done
}

main