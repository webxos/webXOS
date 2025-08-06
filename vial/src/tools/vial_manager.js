/**
 * Vial management logic for Vial MCP Controller
 * Dependencies: json-schema, /vial/schemas/vial.schema.json
 * Handles vial creation and validation
 * Rebuild: Ensure /vial/schemas/vial.schema.json exists
 */
const fs = require('fs');
const path = require('path');
const Ajv = require('ajv');
const ajv = new Ajv();
const vialSchema = JSON.parse(fs.readFileSync(path.join(__dirname, '../schemas/vial.schema.json')));

function validateVial(vialData) {
  try {
    const validate = ajv.compile(vialSchema);
    const valid = validate(vialData);
    if (!valid) throw new Error(`Vial validation failed: ${JSON.stringify(validate.errors)}`);
    return true;
  } catch (err) {
    console.error(`[ERROR] Vial Validation: ${err.message}`);
    return false;
  }
}

function createVial(id, code, training) {
  try {
    const vialData = {
      id,
      code: { js: code },
      training,
      status: 'running',
      latencyHistory: [Math.random() * 100],
      filePath: `/vial/uploads/vial${id}.js`,
      createdAt: new Date().toISOString(),
      codeLength: code.length
    };
    if (!validateVial(vialData)) throw new Error('Invalid vial data');
    fs.writeFileSync(path.join(__dirname, `../uploads/vial${id}.js`), code);
    return vialData;
  } catch (err) {
    console.error(`[ERROR] Create Vial: ${err.message}`);
    throw err;
  }
}

module.exports = { createVial, validateVial };

// Rebuild Instructions: Place in /vial/src/tools/. Install dependency: `npm install ajv`. Ensure /vial/schemas/vial.schema.json exists. Run Troubleshoot in vial.html to check for errors.
