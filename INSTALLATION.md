# Installing Webxos

**Webxos** creates standalone HTML apps using native JavaScript, WebGL, Three.js, and WebAssembly (WASM). Apps like those on [webxos.netlify.app](https://webxos.netlify.app) are single `.html` files, designed for decentralized, lightweight performance. This guide helps you set up to build or test Webxos apps.

## What You Need

- **Browser**: Chrome 126 (supports WebGL 2.0, WASM).
- **Git**: To get the code ([git-scm.com](https://git-scm.com)).
- **Node.js & npm**: Version 18.x/9.x for tools ([nodejs.org](https://nodejs.org)).
- **Optional**: AI tools (Grok, ChatGPT, Claude, Cursor) for code generation.

## Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/webxos/webxos.git
   cd webxos
   ```

2. **Install Tools**:
   ```bash
   npm install
   ```

3. **Run an App**:
   - Add your `.html` file to `apps/` (e.g., `apps/MyApp.html`).
   - Serve locally:
     ```bash
     npm run dev
     ```
   - Open `http://localhost:3000/apps/MyApp.html` in Chrome.
   - Or, open the `.html` file directly in Chrome (no server needed).

## Troubleshooting

- **Tools Fail**: Run `npm install --force`.
- **WebGL Issues**: Check `chrome://gpu` for WebGL 2.0 support.
- **AI Code Errors**: Refine prompts (see [USAGE.md](USAGE.md)).
- Ask for help in [GitHub Issues](https://github.com/webxos/webxos/issues).

## Next Steps

Check [USAGE.md](USAGE.md) to create your own `.html` app.
