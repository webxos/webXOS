/**
 * Agent management for Vial MCP Controller
 * Dependencies: ajv, axios, /vial/schemas/agent.schema.json
 * Handles agent registration, status checks, and AI model connectivity
 * Rebuild: Install dependencies with `npm install ajv axios`
 */
const fs = require('fs');
const path = require('path');
const Ajv = require('ajv');
const axios = require('axios');
const ajv = new Ajv();
const agentSchema = JSON.parse(fs.readFileSync(path.join(__dirname, '../schemas/agent.schema.json')));
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '../mcp.json')));

function validateAgent(agentData) {
  try {
    const validate = ajv.compile(agentSchema);
    const valid = validate(agentData);
    if (!valid) throw new Error(`Agent validation failed: ${JSON.stringify(validate.errors)}`);
    return true;
  } catch (err) {
    console.error(`[ERROR] Agent Validation: ${err.message}`);
    return false;
  }
}

function registerAgent(id, name, endpoints, capabilities) {
  try {
    const agentData = {
      id,
      name,
      endpoints,
      capabilities,
      status: 'active',
      lastPing: new Date().toISOString()
    };
    if (!validateAgent(agentData)) throw new Error('Invalid agent data');
    return agentData;
  } catch (err) {
    console.error(`[ERROR] Register Agent: ${err.message}`);
    throw err;
  }
}

function updateAgentStatus(agentId, status) {
  try {
    const agentData = { id: agentId, status, lastPing: new Date().toISOString() };
    if (!validateAgent({ ...agentData, name: 'temp', endpoints: [], capabilities: { type: 'temp' } })) {
      throw new Error('Invalid agent status update');
    }
    return agentData;
  } catch (err) {
    console.error(`[ERROR] Update Agent Status: ${err.message}`);
    throw err;
  }
}

async function connectToAIModel(modelName, input) {
  try {
    const model = config.ai_models.find(m => m.name.toLowerCase() === modelName.toLowerCase());
    if (!model) throw new Error(`Model ${modelName} not configured in mcp.json`);
    const response = await axios.post(`${model.endpoint}/mcp/query`, { input }, { timeout: 10000 });
    return response.data;
  } catch (err) {
    console.error(`[ERROR] AI Model ${modelName} Connection: ${err.message}`);
    throw err;
  }
}

module.exports = { validateAgent, registerAgent, updateAgentStatus, connectToAIModel };

// Rebuild Instructions: Place in /vial/src/tools/. Install dependencies: `npm install ajv axios`. Ensure /vial/schemas/agent.schema.json and /vial/mcp.json exist. Run Troubleshoot in vial.html to check for errors.
