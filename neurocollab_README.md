# NEUROCOLLAB v1.1  
**Collaborative Multi-Agent Neural Sandbox**

A playful, retro-futuristic single-page web application where four stylized neural agents collaborate in real time — chatting, voting, painting, writing notes, and building simple 3D scenes.

Four agents with different personalities:

- **NEURO-A** — creative  
- **NEURO-B** — analytical  
- **NEURO-C** — technical  
- **NEURO-D** — visionary  

They can:

- discuss ideas in a shared CLI-style chat (with upvotes/downvotes)  
- collaboratively draw abstract paintings (HTML Canvas)  
- write stories/poems/ideas in a shared notepad  
- build primitive 3D shapes (fake 3D with Canvas 2D projection)  
- train / evolve / crossover simulated neural parameters  
- export everything as structured JSON (inbox-only or "full brain")

## Features

- Draggable & resizable retro windows  
- Dark mode ↔ light mode toggle  
- Fullscreen support  
- Simulated training progress bar  
- Export full session data → ready for Hugging Face / datasets  
- Pure HTML + CSS + vanilla JavaScript (no build step, no dependencies)

## Keyboard shortcuts

| Key               | Action                           |
|-------------------|----------------------------------|
| **F1**            | Start agent collaboration        |
| **F2**            | Stop collaboration               |
| **Ctrl + D**      | Toggle dark mode                 |
| **Ctrl + F**      | Toggle fullscreen                |
| **F5**            | Show export modal                |
| **Esc**           | Close export modal               |
| **/****           | Type commands in CLI (`/help`)   |

## Planned / ideas

- Replace canvas 3D projection → real **Three.js** scene  
- WebSocket / local peer collaboration (multiple browsers)  
- Real tiny ML models via transformers.js or ONNX  
- Agent avatars / animated typing indicators  
- Sound reactivity / generative music  
- Save/load named sessions to IndexedDB

## License

MIT

# webXOS
webxos.netlify.app
