/**
 * Log handling for Vial MCP Controller
 * Dependencies: lz-string, /vial/schemas/log.schema.json
 * Manages log compression and validation
 * Rebuild: Ensure /vial/schemas/log.schema.json exists
 */
const fs = require('fs');
const path = require('path');
const Ajv = require('ajv');
const LZString = require('lz-string');
const ajv = new Ajv();
const logSchema = JSON.parse(fs.readFileSync(path.join(__dirname, '../schemas/log.schema.json')));

function validateLog(logData) {
  try {
    const validate = ajv.compile(logSchema);
    const valid = validate(logData);
    if (!valid) throw new Error(`Log validation failed: ${JSON.stringify(validate.errors)}`);
    return true;
  } catch (err) {
    console.error(`[ERROR] Log Validation: ${err.message}`);
    return false;
  }
}

function compressLog(logData) {
  try {
    if (!validateLog(logData)) throw new Error('Invalid log data');
    return LZString.compressToUTF16(JSON.stringify(logData));
  } catch (err) {
    console.error(`[ERROR] Log Compression: ${err.message}`);
    throw err;
  }
}

function decompressLog(compressedLog) {
  try {
    const decompressed = LZString.decompressFromUTF16(compressedLog);
    const logData = JSON.parse(decompressed);
    if (!validateLog(logData)) throw new Error('Invalid decompressed log');
    return logData;
  } catch (err) {
    console.error(`[ERROR] Log Decompression: ${err.message}`);
    throw err;
  }
}

module.exports = { compressLog, decompressLog, validateLog };

// Rebuild Instructions: Place in /vial/src/tools/. Install dependencies: `npm install ajv lz-string`. Ensure /vial/schemas/log.schema.json exists. Run Troubleshoot in vial.html to check for errors.
