# WEBXOS — Chatbot (version 2)

**A single-file, zero-dependency, Markdown-only personal knowledge base chatbot**  
Train an AI assistant directly in your browser using your own `.md` files. No backend, no API keys, fully offline.

## Features

- **Zero dependencies** – pure HTML + CSS + vanilla JS (~25 KB total)
- Upload any number of Markdown (`.md`) files
- Automatic heading-based section extraction
- Builds a consolidated **Knowledge Pack** (single Markdown file)
- Simple hybrid retrieval: exact QA match → title similarity → fallback guidance
- Real-time training progress + topic badges
- Download trained knowledge pack as `webxos-knowledge-pack.md`
- Chat interface with Markdown-aware responses
- Fully works offline after first load (save as HTML and use forever)

## How It Works

1. **Train tab** → Drag & drop or select `.md` files  
2. Click **"Parse & Build Pack"** → headings become topics, content becomes answers
3. Switch to **Chat** → Ask natural questions about your documents
4. The bot matches your question against:
   - Generated Q&A ("tell me about X")
   - Heading similarity scoring
   - Fallback with relevant topic suggestions

No vector DB, no embeddings — just fast string/token matching.

## Use Cases

- Personal knowledge base from Obsidian/PKM notes
- Documentation chatbot for open-source projects
- Offline company wiki assistant
- Student notes → instant Q&A tool
- Private RAG prototype (100% in-browser)
