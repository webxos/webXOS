# WEBXOS — Chatbot (version 2)

**A single-file, zero-dependency, Markdown-only personal knowledge base chatbot**  
Train an AI assistant directly in your browser using your own `.md` files. No backend, no API keys, fully offline.

![[CHATBOT BANNER](https://github.com/webxos/webXOS/blob/main/assets/chatbot.jpeg)](https://github.com/webxos/webXOS/blob/main/assets/chatbot.jpeg)  

# UNDER DEVELOPMENT — UPDATED DECEMBER 1, 2025

**Local, private, offline-first AI chatbot with full markdown knowledge-base + sandboxed JavaScript/Python script execution**

*webxos.netlify.app/chatbot*

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
![Pyodide](https://img.shields.io/badge/Powered_by-Pyodide-39FF14)  
![No backend](https://img.shields.io/badge/No_backend-100%25_local-blue)

**WARNING KNOWN ERRORS**

1: Some of the js/py catalog is broken the scripts do not work (calculator etc)
2. Fix coming soon for all script library templates will work.

## Features

- **Full Knowledge Pack Reading** – Drop any number of `.md` files → the bot reads **everything**, not just tiny chunks
- **Live editing** of knowledge packs (changes affect answers instantly)
- **Sandboxed script execution** – JavaScript (via browser) + Python (via Pyodide) inside code blocks
- **Auto-capability detection** – `// capabilities: math, calculate` → bot automatically runs the right script
- **Progressive disclosure** – content is shown in readable paragraphs with clear headings
- **Export / Import everything** in a single markdown file
- **100% offline** – works without internet after first load (Pyodide is cached)
- **No server, no tracking, no accounts**
- **Lightweight script execution (js and py skulpt/pyodide)**
  
## Usage

*Follow the in app guide for proper usage*

### Context Window

- Click **Context Window** button → add markdown files with context for the chatbot
- Click any pack → edit the markdown on the right
- Changes are saved instantly and affect suggestions immediately to start working with chatbot
