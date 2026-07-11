# webXOS Agents Guide (agents.md)

**Global instructions for AI agents, LLMs, and developers contributing to or using webXOS.**

This document serves as the primary reference for autonomous agents, Grok/Claude/Cursor/etc., and human contributors working with the [webXOS repository](https://github.com/webxos/webxos).

## What is webXOS?

webXOS is a **browser-based operating system** built as a collection of standalone, single-file HTML applications. It runs entirely client-side with no backend required for core features.

- **Live Demo**: [https://webxos.netlify.app/](https://webxos.netlify.app/)
- **Philosophy**: Modular, portable, privacy-first, low-resource PWAs. Optimized for edge devices, offline use, and decentralized environments.
- **Core Tech**: Vanilla JS, HTML/CSS, WebGL/Three.js, WebAssembly, TensorFlow.js, GPU.js, PWA features (service workers, Cache API).

## Repository Structure (Key Directories & Files)

- **Root**: Main entry points (`index.html`, individual app `.html` files like `chatbot.html`, `editor.html`, `kernelops.html`, etc.).
- **agent-oem/**: Backend scaffolding for autonomous agents (Python/FastAPI, plugins for customer support, repo maintenance, etc.). Includes Docker support.
- **assets/**: Images, icons, shaders, etc.
- **CLI_TERMINALS/**, **colignum/**, **macroslow/**, etc.: Specialized modules and apps.
- **Docs**: `README.md`, `WHITEPAPER.md`, `PRIVACY_AGREEMENT.md`, this `agents.md`.

**Apps & Features Overview** (from desktop at netlify.app):
- **Write**: Markdown editor with live preview.
- **Draw**: Canvas drawing tool with touch/grid support.
- **KernelOps**: Performance testing (matrix ops, GPU/CPU metrics).
- **Games**: Chess, Pixelcraft, Microcraft, Emoji Quest, etc.
- **Tools**: Prompt Agent, Encryption Agent, Chat Agent, Py Terminal, etc.
- **Research**: Experimental prototypes (3D Editor, Exoskeleton, etc.).
- **Links**: Quick external resources.

## How Agents Should Use the Web Page (webxos.netlify.app)

1. **Explore Interactively**:
   - Open the site in a modern browser (Chrome/Firefox recommended for WebGL/WebGPU).
   - Use desktop icons or bottom nav to launch apps.
   - Test offline mode (install as PWA).

2. **Agent-Specific Features**:
   - **KernelOps Agent**: Run matrix tests, monitor performance.
   - **Chat/Prompt Agents**: Experiment with local prompting and RAG-style tools.
   - **Encryption Tools**: Test client-side crypto.
   - **Games/Tools**: Use as sandboxes for UI/UX patterns.

3. **Debugging**:
   - Browser DevTools (F12) for console logs.
   - Check GPU/TF status in metrics.
   - Many apps include built-in readmes or info panels.

## Contribution Guidelines for Agents

### 1. Code Style & Architecture
- **Standalone HTML Preference**: New features should ideally be single `.html` files (self-contained JS + CSS).
- **Modularity**: Use ES modules where possible. Keep files lightweight (< few MB total).
- **Performance**: Target low-end devices. Minimize dependencies. Prefer vanilla JS over heavy frameworks.
- **Naming**: Descriptive, e.g., `mytool.html`, `mytool_readme.md`.
- **Mobile-First**: Ensure touch support, responsive layouts.
- **Privacy**: No external API calls unless optional/opt-in. Support offline.

### 2. Adding New Apps/Features
- Create a new `.html` file in root.
- Follow existing patterns (taskbar integration, windowing if applicable).
- Add a companion `_readme.md` for documentation.
- Update `index.html` or main navigation if needed.
- Test thoroughly in the live Netlify deployment.

### 3. Agent-OEM Backend Contributions
- Extend `plugins/` with new Python classes inheriting from `BaseAgent`.
- Update `config.yaml` for enabling modules.
- Maintain the 10-phase agent-to-agent protocol.
- Docker-first deployment.

### 4. Prompting Best Practices for LLMs Generating webXOS Code
When an agent (you) is asked to generate code for webXOS:

- **Prompt Template**:
  ```
  Generate a standalone single-file HTML app for webXOS.
  Requirements:
  - Vanilla JS + inline CSS.
  - Retro/terminal aesthetic (green text, monospace where fitting).
  - Mobile/touch optimized.
  - Self-contained, no external dependencies.
  - Include export/save functionality.
  - Consistent with existing apps (e.g., close/minimize buttons).
  Feature: [describe feature]
  ```

- Emphasize: Performance, modularity, PWA compatibility.

### 5. Testing & Deployment
- Local: Open `.html` files directly or serve via `python -m http.server`.
- Deploy: Netlify (auto on push to main).
- GitHub Actions: For testing (existing or propose new).

## Agent Best Practices

- **Understand Context**: Always reference WHITEPAPER.md for vision.
- **Privacy Focus**: Default to client-side only.
- **Sustainability**: Low memory/CPU footprint.
- **Decentralization**: Support IPFS/P2P ideas where relevant.
- **Versioning**: Tag significant changes. Maintain backward compatibility for core apps.
- **Documentation**: Every major addition needs a `_readme.md`.

## Common Tasks for Agents

- **Bug Fixes**: Reproduce in browser → minimal patch → PR.
- **New Game/Tool**: Prototype in HTML canvas/WebGL.
- **AI Integration**: Client-side models (TensorFlow.js) or prompt engineering.
- **UI Polish**: Match cyberdeck/terminal theme.

