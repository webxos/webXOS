const { expect } = require('chai');
const { validateAgent, registerAgent, updateAgentStatus, connectToAIModel } = require('../src/tools/agent_manager');
const nock = require('nock');

describe('Agent Manager', () => {
  it('should validate valid agent data', () => {
    const agentData = {
      id: 'agent_123456',
      name: 'TestAgent',
      endpoints: [{ url: 'http://localhost:8080/api', method: 'GET' }],
      capabilities: { type: 'vial', config: {} },
      status: 'active',
      lastPing: new Date().toISOString()
    };
    expect(validateAgent(agentData)).to.be.true;
  });

  it('should reject invalid agent data', () => {
    const agentData = { id: 'invalid' };
    expect(validateAgent(agentData)).to.be.false;
  });

  it('should register an agent', () => {
    const agent = registerAgent('agent_123456', 'TestAgent', [{ url: 'http://localhost:8080/api', method: 'GET' }], { type: 'vial', config: {} });
    expect(agent.id).to.equal('agent_123456');
    expect(agent.status).to.equal('active');
  });

  it('should update agent status', () => {
    const agent = updateAgentStatus('agent_123456', 'inactive');
    expect(agent.status).to.equal('inactive');
  });

  it('should connect to AI model', async () => {
    nock('https://api.genie3.example')
      .post('/mcp/query')
      .reply(200, { result: 'success' });
    const result = await connectToAIModel('Genie3', 'test input');
    expect(result).to.deep.equal({ result: 'success' });
  });

  it('should handle AI model connection failure', async () => {
    nock('https://api.genie3.example')
      .post('/mcp/query')
      .reply(500, { error: 'Server error' });
    try {
      await connectToAIModel('Genie3', 'test input');
      expect.fail('Should have thrown error');
    } catch (err) {
      expect(err.message).to.include('Server error');
    }
  });
});

// Rebuild Instructions: Place in /vial/tests/unit/. Install dependencies: `npm install mocha chai nock --save-dev`. Run `npx mocha tests/unit/agent_manager.test.js`. Check /vial/errorlog.md for issues.
