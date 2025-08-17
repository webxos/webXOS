import { config } from './config.js';
import { VialManager } from './vial-manager.js';
import { ErrorHandler } from './error-handler.js';
import { MCPClient, Gateway } from './index.html'; // Assumes index.html exports these classes

export async function runTests() {
  console.log('Running Vial MCP frontend tests...');
  const client = new MCPClient();
  const gateway = new Gateway();
  const vialManager = new VialManager(client, gateway);
  const errorHandler = new ErrorHandler(client, gateway);

  // Test 1: Environment Detection
  console.log('Test 1: Environment Detection');
  console.assert(client.environment === (window.location.hostname.includes('netlify.app') ? 'production' : window.location.hostname === 'localhost' ? 'local' : 'demo'), `Expected environment ${client.environment}, got ${client.environment}`);

  // Test 2: Mock API Response
  console.log('Test 2: Mock API Response');
  const mockHealth = await client.request('/health');
  console.assert(mockHealth.balance === config.DEFAULT_BALANCE, `Expected balance ${config.DEFAULT_BALANCE}, got ${mockHealth.balance}`);

  // Test 3: Vial Start
  console.log('Test 3: Vial Start');
  await vialManager.startVial('vial1');
  const vial1Status = gateway.vials.find(v => v.id === 'vial1')?.status;
  console.assert(vial1Status === 'Training' || vial1Status === 'Running', `Expected vial1 status Training or Running, got ${vial1Status}`);

  // Test 4: Error Handling
  console.log('Test 4: Error Handling');
  const error = new Error('Invalid token');
  const fallback = errorHandler.handle(error, '/health');
  console.assert(fallback.user_id === config.DEFAULT_USER_ID, `Expected fallback user_id ${config.DEFAULT_USER_ID}, got ${fallback.user_id}`);

  // Test 5: Command Validation
  console.log('Test 5: Command Validation');
  const validCommand = '/auth';
  const invalidCommand = '/invalid';
  console.assert(client.validateCommand(validCommand), `Expected ${validCommand} to be valid`);
  console.assert(!client.validateCommand(invalidCommand), `Expected ${invalidCommand} to be invalid`);

  console.log('Tests completed');
}
