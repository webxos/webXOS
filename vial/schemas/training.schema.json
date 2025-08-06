/**
 * Training logic for Vial MCP Controller
 * Dependencies: /vial/schemas/training.schema.json
 * Manages vial training process
 * Rebuild: Ensure /vial/schemas/training.schema.json exists
 */
const fs = require('fs');
const path = require('path');
const Ajv = require('ajv');
const ajv = new Ajv();
const trainingSchema = JSON.parse(fs.readFileSync(path.join(__dirname, '../schemas/training.schema.json')));

function validateTraining(trainingData) {
  try {
    const validate = ajv.compile(trainingSchema);
    const valid = validate(trainingData);
    if (!valid) throw new Error(`Training validation failed: ${JSON.stringify(validate.errors)}`);
    return true;
  } catch (err) {
    console.error(`[ERROR] Training Validation: ${err.message}`);
    return false;
  }
}

function trainVial(id, input) {
  try {
    const trainingData = {
      id,
      input,
      model: 'default',
      epochs: 5,
      latency: Math.random() * 100,
      codeLength: input.length
    };
    if (!validateTraining(trainingData)) throw new Error('Invalid training data');
    return trainingData;
  } catch (err) {
    console.error(`[ERROR] Train Vial: ${err.message}`);
    throw err;
  }
}

module.exports = { trainVial, validateTraining };

// Rebuild Instructions: Place in /vial/src/tools/. Install dependency: `npm install ajv`. Ensure /vial/schemas/training.schema.json exists. Run Troubleshoot in vial.html to check for errors.
