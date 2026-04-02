#!/bin/bash
# start.sh – RustyClaw v0.6.0 launcher

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}${CYAN}"
cat << 'EOF'
▄▖    ▗     ▜      
▙▘▌▌▛▘▜▘▌▌▛▘▐ ▀▌▌▌▌
▌▌▙▌▄▌▐▖▙▌▙▖▐▖█▌▚▚▘
        ▄▌         
EOF
echo -e "${NC}"

# Check Cargo
if ! command -v cargo &>/dev/null; then
    echo -e "${RED}❌ Cargo not found. Install Rust: https://rustup.rs/${NC}"
    exit 1
fi
echo -e "${GREEN}✅ $(cargo --version)${NC}"

# Ensure directories
mkdir -p src data logs

# Move main.rs if needed
if [ -f "main.rs" ] && [ ! -f "src/main.rs" ]; then
    echo -e "${YELLOW}⚠️  Moving main.rs → src/main.rs${NC}"
    mv main.rs src/main.rs
fi

if [ ! -f "src/main.rs" ]; then
    echo -e "${RED}❌ src/main.rs not found.${NC}"
    exit 1
fi

# Ollama check
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✅ Ollama running — ${MODEL_COUNT} model(s) available${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama not detected at localhost:11434${NC}"
    echo -e "${YELLOW}   Start with: ollama serve${NC}"
    echo -e "${YELLOW}   Pull model: ollama pull qwen2.5:0.5b${NC}"
    echo ""
    read -rp "   Continue anyway? (y/N) " reply; echo
    [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

# Build
BINARY="./target/release/rustyclaw"
FORCE_REBUILD="${1:-}"

needs_build=false
[ ! -f "$BINARY" ] && needs_build=true
[ "src/main.rs" -nt "$BINARY" ] 2>/dev/null && needs_build=true
[ "Cargo.toml"  -nt "$BINARY" ] 2>/dev/null && needs_build=true
[ "$FORCE_REBUILD" = "--rebuild" ] && needs_build=true

if $needs_build; then
    echo -e "${BLUE}🔨 Building RustyClaw (release)…${NC}"
    cargo build --release
    echo -e "${GREEN}✅ Build complete${NC}"
else
    echo -e "${CYAN}⚡ Binary up-to-date (./start.sh --rebuild to force)${NC}"
fi

# Launch
echo ""
echo -e "${GREEN}⚙️  Launching RustyClaw…${NC}"
echo -e "${CYAN}   ESC or /quit to exit  │  /help for commands${NC}"
echo -e "${CYAN}   REST API: http://127.0.0.1:3030/health${NC}"
echo -e "${CYAN}   Persona:  ~/.rustyclaw/bio.md${NC}"
echo -e "${CYAN}   Files:    ~/.rustyclaw/data/ (Git repo)${NC}"
echo ""
exec "$BINARY"
