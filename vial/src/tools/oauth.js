/**
 * OAuth handling for Vial MCP Controller
 * Dependencies: jsonwebtoken, /vial/schemas/oauth.schema.json
 * Manages token generation and validation
 * Rebuild: Ensure /vial/schemas/oauth.schema.json and /vial/mcp.json exist
 */
const fs = require('fs');
const path = require('path');
const jwt = require('jsonwebtoken');
const Ajv = require('ajv');
const ajv = new Ajv();
const oauthSchema = JSON.parse(fs.readFileSync(path.join(__dirname, '../schemas/oauth.schema.json')));
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '../mcp.json')));

function validateOAuthToken(tokenData) {
  try {
    const validate = ajv.compile(oauthSchema);
    const valid = validate(tokenData);
    if (!valid) throw new Error(`OAuth validation failed: ${JSON.stringify(validate.errors)}`);
    return true;
  } catch (err) {
    console.error(`[ERROR] OAuth Validation: ${err.message}`);
    return false;
  }
}

function generateToken(clientId, scopes) {
  try {
    const tokenData = {
      client_id: clientId,
      scopes,
      exp: Math.floor(Date.now() / 1000) + 3600,
      iat: Math.floor(Date.now() / 1000)
    };
    if (!validateOAuthToken(tokenData)) throw new Error('Invalid token data');
    return jwt.sign(tokenData, process.env.OAUTH_CLIENT_SECRET || config.oauth.client_secret);
  } catch (err) {
    console.error(`[ERROR] Token Generation: ${err.message}`);
    throw err;
  }
}

function verifyToken(token) {
  try {
    const decoded = jwt.verify(token, process.env.OAUTH_CLIENT_SECRET || config.oauth.client_secret);
    if (!validateOAuthToken(decoded)) throw new Error('Invalid token');
    return decoded;
  } catch (err) {
    console.error(`[ERROR] Token Verification: ${err.message}`);
    throw err;
  }
}

module.exports = { generateToken, verifyToken, validateOAuthToken };

// Rebuild Instructions: Place in /vial/src/tools/. Install dependencies: `npm install jsonwebtoken ajv`. Ensure /vial/schemas/oauth.schema.json and /vial/mcp.json exist. Run Troubleshoot in vial.html to check for errors.
