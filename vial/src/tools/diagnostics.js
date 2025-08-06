/**
 * System diagnostics for Vial MCP Controller
 * Dependencies: fs, path
 * Checks server health and file presence
 * Rebuild: Ensure /vial/mcp.json exists
 */
const fs = require('fs');
const path = require('path');

function runDiagnostics() {
  const issues = [];
  try {
    const configPath = path.join(__dirname, '../mcp.json');
    if (!fs.existsSync(configPath)) issues.push({ message: 'mcp.json missing', analysis: 'Check /vial/mcp.json' });
    
    const staticFiles = [
      'redaxios.min.js', 'lz-string.min.js', 'mustache.min.js', 'dexie.min.js',
      'jwt-decode.min.js', 'sql-wasm.wasm', 'worker.js', 'icon.png', 'manifest.json'
    ];
    staticFiles.forEach(file => {
      if (!fs.existsSync(path.join(__dirname, '../static', file))) {
        issues.push({ message: `Missing file: ${file}`, analysis: `Check /vial/static/${file} or download from cdns.txt` });
      }
    });

    const dbPath = process.env.DB_PATH || './vial.db';
    if (!fs.existsSync(dbPath)) issues.push({ message: 'Database file missing', analysis: 'Check DB_PATH in .env' });

    return { issues };
  } catch (err) {
    console.error(`[ERROR] Diagnostics: ${err.message}`);
    issues.push({ message: `Diagnostics Error: ${err.message}`, analysis: 'Check /vial/src/diagnostics.js' });
    return { issues };
  }
}

module.exports = { runDiagnostics };

// Rebuild Instructions: Place in /vial/src/tools/. Ensure /vial/mcp.json and /vial/static/ files exist. Run Troubleshoot in vial.html to check for errors.
