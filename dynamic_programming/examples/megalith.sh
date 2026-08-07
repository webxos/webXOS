#!/usr/bin/env bash
# Minimal UNIXAI-style skeleton – demonstrates the modular pattern.
set -euo pipefail

MODEL="${UNIXAI_MODEL:-llama3.2}"
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

echo "Minimal Megalith Agent ready (model=$MODEL)"
echo "Type /help or any prompt. 'exit' to quit."

while true; do
  printf "You> "
  read -r INPUT || break
  case "$INPUT" in
    exit|quit|q) break ;;
    /help)
      echo "Commands: /help, /status, exit"
      echo "Any other text is sent to the model (stub)."
      ;;
    /status)
      echo "Ollama URL: $OLLAMA_URL"
      echo "Model: $MODEL"
      ;;
    *)
      echo "AI> [stub] You said: $INPUT"
      # In a real megalith this would call ollama /api/generate
      ;;
  esac
done
echo "Session ended."
