# Contributing to Webxos

**Webxos** builds lightweight, standalone HTML apps for the decentralized web, using native JavaScript, WebGL, Three.js, and WebAssembly (WASM). Apps like those on [webxos.netlify.app](https://webxos.netlify.app) (e.g., eco-friendly tools, retro games, AI agents like FalseNode@webxos) are single `.html` files, ensuring modularity and performance. Here’s how to contribute a new app.

## How to Contribute

1. **Get the Code**:
   - Clone the repository:
     ```bash
     git clone https://github.com/webxos/webxos.git
     cd webxos
     ```

2. **Create Your App**:
   - Make a single `.html` file in `apps/` (e.g., `apps/MyApp.html`).
   - Include all HTML, CSS, JavaScript, and assets (e.g., WebGL or WASM) inline.
   - Example: Use Three.js for a game or IPFS for P2P file sharing.
   - Test in Chrome 126 (mobile/desktop, online/offline).

3. **Use AI Tools** (Optional):
   - Use Grok, ChatGPT, Claude, or Cursor to generate code. Example prompt:
     ```
     Create a single-page HTML app for a WebGL game, with a neon UI and offline support.
     ```
   - Embed the generated code in your `.html` file.

4. **Submit Your App**:
   - Create a branch:
     ```bash
     git checkout -b app/my-app-name
     ```
   - Commit your `.html` file:
     ```bash
     git add apps/MyApp.html
     git commit -m "Add MyApp.html for WebGL visualization"
     ```
   - Push and open a pull request to [webxos/webxos](https://github.com/webxos/webxos).

## Guidelines

- Keep it simple: One `.html` file, no external frameworks (e.g., React).
- Add meta tags (charset, viewport, description) and a footer (e.g., `MyApp v1.0.0 © 2025 WEBXOS`).
- Optimize for low-end devices (≥320px) and offline use (Cache API).
- Check [USAGE.md](USAGE.md) for example code.

## Community

- Share on [@webxos](https://x.com/webxos).
- Join [GitHub Discussions](https://github.com/webxos/webxos/discussions).
- Explore [webxos.netlify.app](https://webxos.netlify.app).

Thank you for building with Webxos’s minimal, decentralized framework!
