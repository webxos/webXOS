# WEBXOS — Chatbot (version 2)

**A single-file, zero-dependency, Markdown-only personal knowledge base chatbot**  
Train an AI assistant directly in your browser using your own `.md` files. No backend, no API keys, fully offline.

![[CHATBOT BANNER](https://github.com/webxos/webXOS/blob/main/assets/chatbot.jpeg)](https://github.com/webxos/webXOS/blob/main/assets/chatbot.jpeg)  

# WEBXOS CHATBOT — UPDATES NOVEMBER 25, 2025

**Local, private, offline-first AI chatbot with full markdown knowledge-base + sandboxed JavaScript/Python script execution**

*webxos.netlify.app/chatbot*

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
![Pyodide](https://img.shields.io/badge/Powered_by-Pyodide-39FF14)  
![No backend](https://img.shields.io/badge/No_backend-100%25_local-blue)

## Features

- **Full Knowledge Pack Reading** – Drop any number of `.md` files → the bot reads **everything**, not just tiny chunks
- **Live editing** of knowledge packs (changes affect answers instantly)
- **Sandboxed script execution** – JavaScript (via browser) + Python (via Pyodide) inside code blocks
- **Auto-capability detection** – `// capabilities: math, calculate` → bot automatically runs the right script
- **Progressive disclosure** – content is shown in readable paragraphs with clear headings
- **Export / Import everything** in a single markdown file
- **100% offline** – works without internet after first load (Pyodide is cached)
- **No server, no tracking, no accounts**

## Quick Start (0-click)

1. Save the file as `webxos-chatbot.html`
2. Double-click it → opens in your browser
3. Drag & drop any `.md` files onto the **Train** tab
4. Click **Auto-train**
5. Go to **Chat** and ask anything → the bot now knows everything you just fed it

## Detailed Usage

### 1. Adding Knowledge

- Go to **Train** tab
- Drag & drop one or many markdown files (`.md`, `.markdown`, `.mdx`)
- Click **Auto-train**
- Each file becomes a separate “Knowledge Pack”

### 2. Editing Knowledge (live)

- Click **Knowledge Packs** button → modal opens
- Click any pack → edit the markdown on the right
- Changes are saved instantly and affect answers immediately
