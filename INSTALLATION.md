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

### Common Installation Issues

#### 1. Repository Access Issues
**Problem**: `Repository not found` or permission errors when cloning.

**Solutions**:
- Verify you have access to the repository
- Use HTTPS instead of SSH: `git clone https://github.com/webxos/webxos.git`
- For private repos, set up authentication:
  - **Personal Access Token**: [Create one on GitHub](https://github.com/settings/tokens) → Use as password when cloning
  - **SSH Keys**: [Set up SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) for password-free access

#### 2. npm Install Failures
**Problem**: `npm install` fails or shows dependency errors.

**Solutions**:
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then reinstall:
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```
- Try with force flag: `npm install --force`
- Ensure you're using Node.js 18.x: `node --version`

#### 3. Package.json Errors
**Problem**: JSON parse errors or invalid package.json.

**Solutions**:
- Validate JSON syntax at [jsonlint.com](https://jsonlint.com)
- Check for:
  - Missing commas or brackets
  - Trailing commas (not allowed in JSON)
  - Correct quote usage (double quotes only)

#### 4. Module Not Found Errors
**Problem**: `Cannot find module` errors when running apps.

**Solutions**:
- Verify the "main" entry in package.json points to the correct file
- Check that all dependencies are installed: `npm install`
- Ensure file paths are correct (case-sensitive on Linux/Mac)

#### 5. Git Push Failures
**Problem**: Authentication errors when pushing changes.

**Solutions**:
- **Using HTTPS**: Generate a [Personal Access Token](https://github.com/settings/tokens) and use it as your password
- **Using SSH**: Set up [SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
- Update remote URL if needed:
  ```bash
  git remote set-url origin https://github.com/webxos/webxos.git
  ```

#### 6. WebGL Issues
**Problem**: Apps don't render or show WebGL errors.

**Solutions**:
- Check browser support: Visit `chrome://gpu` in Chrome
- Ensure WebGL 2.0 is enabled
- Update graphics drivers
- Try a different browser (Chrome, Firefox, Edge)

#### 7. Server Won't Start
**Problem**: `npm run dev` fails or port already in use.

**Solutions**:
- Check if port 3000 is already in use
- Kill existing process:
  - **Windows**: `netstat -ano | findstr :3000` then `taskkill /PID <PID> /F`
  - **Mac/Linux**: `lsof -ti:3000 | xargs kill -9`
- Use a different port: Edit package.json scripts to use `--port 3001`

### Getting More Help
- Check existing [GitHub Issues](https://github.com/webxos/webxos/issues)
- Create a new issue with:
  - Error message (full text)
  - Your OS and Node.js version
  - Steps to reproduce
- Join discussions on [@webxos](https://x.com/webxos)

## Next Steps
Check [USAGE.md](USAGE.md) to create your own `.html` app.
