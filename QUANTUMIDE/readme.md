# WebXOS 2025 Quantum IDE (COMING SOON)

**WebXOS Quantum IDE** is a lightweight, open-source development environment for quantum computing, built for model context protocol (MCP) workflows. It features a single-page front-end (`quantumide.html`, ~15KB minified) with inline CSS/JavaScript, powered by **Q.js** (MIT license) for client-side quantum circuit simulations (up to ~20 qubits) and a **Flask/QuTiP** backend (Apache license) for advanced quantum dynamics with diagnostics (e.g., fidelity, noise, up to ~30 qubits). The project is deployable as a static SPA on **Netlify** (`webxos.netlify.app`) and integrates with **GitHub Copilot** for AI-assisted linting and code fixes via GitHub Actions.

All components are free, open-source, and hosted on GitHub for cloning, contribution, and deployment.

## Features
- **Front-End**: Single HTML file with Q.js for drag-and-drop quantum circuit editing, simulation, and visualization (probabilities, Bloch spheres).
- **Backend**: Flask server with QuTiP for advanced simulations (noisy dynamics, fidelity metrics) via API endpoints (`/api/simulator`, `/api/settings`).
- **Linting & Diagnostics**: GitHub Actions lints HTML, inline CSS/JS, Python, and YAML, posting Copilot-compatible PR comments for automated fixes.
- **Deployment**: Static SPA on Netlify; backend via Docker for scalability.
- **Extensibility**: Add Qiskit/Cirq via `requirements.txt` for alternative quantum backends.
- **License**: MIT/Apache, fully open-source.

## Repository Structure
| File Path | Purpose |
|-----------|---------|
| `.github/workflows/lint.yml` | Lints all files, extracts inline CSS/JS, posts diagnostics for Copilot. |
| `.github/dependabot.yml` | Auto-updates Python/Action dependencies for security/performance. |
| `quantum-ide/quantumide.html` | Self-contained SPA with Q.js (quantum sims) and Three.js (visuals). |
| `quantum-ide/server.py` | Flask backend with API endpoints for simulations/settings. |
| `quantum-ide/qutip_sim.py` | QuTiP module for advanced quantum sims (Bell state, fidelity). |
| `quantum-ide/config.yaml` | YAML config for API keys and quantum parameters. |
| `quantum-ide/requirements.txt` | Python dependencies (Flask, QuTiP, etc.). |
| `quantum-ide/Dockerfile` | Dockerizes backend for deployment. |
| `quantum-ide/netlify.toml` | Configures Netlify for SPA routing. |
| `quantum-ide/.gitignore` | Ignores build artifacts/logs. |

## Prerequisites
- **Git**: Clone the repository.
- **Python 3.11**: Run the backend.
- **Docker**: Optional, for backend deployment.
- **Node.js**: Optional, for local linting (Actions handles CI linting).
- **GitHub Account**: For Copilot and repository management.
- **Netlify Account**: For front-end deployment.
- **VS Code + GitHub Copilot**: For AI-assisted coding/diagnostics.

## Installation and Setup
### 1. Clone the Repository
```bash
git clone https://github.com/webxos/webXOS.git
cd webXOS/quantum-ide
```

### 2. Install Backend Dependencies
```bash
pip install -r quantum-ide/requirements.txt
```
Installs Flask, QuTiP, PyYAML, NumPy, Flask-CORS.

### 3. Configure the Backend
Edit `quantum-ide/config.yaml`:
```yaml
api:
  key: "your-api-key-here"  # Set a secure API key
quantum:
  qubits: 20  # Max qubits for sims
qutip:
  noise_level: 0.01  # Noise for diagnostics
```

### 4. Run the Backend
```bash
python quantum-ide/server.py
```
Runs on `http://localhost:5000`. Endpoints:
- `GET/POST /api/simulator`: Status or run QuTiP sim (POST JSON: `{"circuit": "bell"}`).
- `GET /api/settings`: API configuration.

### 5. Run the Front-End
Serve `quantum-ide/quantumide.html`:
```bash
python -m http.server 8000
```
Open `http://localhost:8000/quantumide.html`. Features:
- Q.js client-side sims (Simulator tab, up to 20 qubits).
- Backend API calls for QuTiP sims.

### 6. Deploy Backend with Docker
```bash
docker build -t quantum-ide-backend -f quantum-ide/Dockerfile .
docker run -p 5000:5000 quantum-ide-backend
```

### 7. Deploy Front-End to Netlify
1. Push to GitHub:
   ```bash
   git add . && git commit -m "Deploy Quantum IDE" && git push origin main
   ```
2. In Netlify:
   - Connect GitHub repo.
   - Build Command: (empty).
   - Publish Directory: `.`.
   - Deploy to `webxos.netlify.app/quantumide.html`.
   - `netlify.toml` ensures SPA routing.

## Quantum Simulations
- **Client-Side (Q.js)**:
  - Simulator tab: Drag-and-drop circuit editor (e.g., Bell state: H, CNOT).
  - Click "Run Q.js Sim" for probabilities/Bloch spheres.
  - Lightweight, ~20 qubits max.
- **Server-Side (QuTiP)**:
  - POST circuits to `/api/simulator` for noisy sims and diagnostics (fidelity).
  - Supports ~30 qubits, extendable via `qutip_sim.py`.

## Linting and Diagnostics
Run locally:
```bash
npm install -g htmlhint eslint stylelint stylelint-config-standard
pip install yamllint flake8 black mypy
htmlhint quantum-ide/quantumide.html
yamllint quantum-ide/config.yaml
flake8 quantum-ide/*.py
black --check quantum-ide/*.py
mypy quantum-ide/*.py
```
GitHub Actions (`.github/workflows/lint.yml`) runs on push/PR, posts diagnostics for Copilot.

## GitHub Copilot Integration
- **Setup**: Install Copilot in VS Code, enable in GitHub repo settings.
- **Diagnostics**: Actions post lint errors as PR comments. Copilot suggests fixes (e.g., "Fix Flake8 in server.py").
- **Usage**: Ask Copilot:
  - "Optimize Bell circuit in qutip_sim.py."
  - "Add Qiskit to requirements.txt."
- **Security**: Enable Copilot security filters to scan for vulnerabilities.

## Extending the Project
- **Add Quantum Libraries**: Update `requirements.txt` with Qiskit/Cirq:
  ```bash
  echo "qiskit==1.2.0" >> quantum-ide/requirements.txt
  pip install -r quantum-ide/requirements.txt
  ```
- **Custom Circuits**: Extend `qutip_sim.py` for new circuits (e.g., Grover’s).
- **Diagnostics**: Add metrics (e.g., entanglement entropy) in `qutip_sim.py`.

## License
MIT License. All components (Q.js, QuTiP, Flask, etc.) are open-source (MIT/Apache).

## Resources
- [Q.js Docs](https://github.com/stewdio/q.js)
- [QuTiP Docs](https://qutip.org)
- [Netlify Docs](https://docs.netlify.com)
- [GitHub Copilot](https://docs.github.com/copilot)

Contribute on GitHub: Fork, branch, PR. Use Copilot for code reviews!
