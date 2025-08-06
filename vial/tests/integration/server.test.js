const { expect } = require('chai');
const request = require('supertest');
const jwt = require('jsonwebtoken');
const app = require('../src/server');

describe('MCP Server', () => {
  const secret = process.env.OAUTH_CLIENT_SECRET || 'your_client_secret';
  const token = jwt.sign({ scopes: ['vial:read', 'vial:write', 'agent:read', 'agent:write', 'agent:config'] }, secret, { expiresIn: '1h' });

  it('should authenticate with valid token', async () => {
    const res = await request(app).post('/mcp/auth').send({ token: 'anonymous' });
    expect(res.status).to.equal(200);
    expect(res.body).to.have.property('token');
  });

  it('should reject invalid token input', async () => {
    const res = await request(app).post('/mcp/auth').send({ token: '' });
    expect(res.status).to.equal(400);
    expect(res.body.error).to.equal('Invalid input');
  });

  it('should register an agent', async () => {
    const res = await request(app)
      .post('/mcp/register')
      .set('Authorization', `Bearer ${token}`)
      .send({
        id: 'agent_123456',
        name: 'TestAgent',
        endpoints: [{ url: 'http://localhost:8080/api', method: 'GET' }],
        capabilities: { type: 'vial', config: {} }
      });
    expect(res.status).to.equal(200);
    expect(res.body.id).to.equal('agent_123456');
  });

  it('should reject invalid agent registration', async () => {
    const res = await request(app)
      .post('/mcp/register')
      .set('Authorization', `Bearer ${token}`)
      .send({ id: 'invalid', name: '' });
    expect(res.status).to.equal(400);
    expect(res.body.error).to.equal('Invalid input');
  });

  it('should list active agents', async () => {
    const res = await request(app).get('/mcp/agents').set('Authorization', `Bearer ${token}`);
    expect(res.status).to.equal(200);
    expect(res.body).to.be.an('array');
  });

  it('should handle agent ping', async () => {
    await request(app)
      .post('/mcp/register')
      .set('Authorization', `Bearer ${token}`)
      .send({
        id: 'agent_123456',
        name: 'TestAgent',
        endpoints: [{ url: 'http://localhost:8080/api', method: 'GET' }],
        capabilities: { type: 'vial', config: {} }
      });
    const res = await request(app)
      .post('/mcp/agent/ping')
      .set('Authorization', `Bearer ${token}`)
      .send({ agentId: 'agent_123456' });
    expect(res.status).to.equal(200);
    expect(res.body.status).to.equal('ok');
  });

  it('should configure an agent', async () => {
    await request(app)
      .post('/mcp/register')
      .set('Authorization', `Bearer ${token}`)
      .send({
        id: 'agent_123456',
        name: 'TestAgent',
        endpoints: [{ url: 'http://localhost:8080/api', method: 'GET' }],
        capabilities: { type: 'vial', config: {} }
      });
    const res = await request(app)
      .post('/mcp/agent/config')
      .set('Authorization', `Bearer ${token}`)
      .send({ agentId: 'agent_123456', config: { model: '3d' } });
    expect(res.status).to.equal(200);
    expect(res.body.config).to.deep.equal({ model: '3d' });
  });

  it('should discover agents and AI models', async () => {
    const res = await request(app).get('/mcp/discover').set('Authorization', `Bearer ${token}`);
    expect(res.status).to.equal(200);
    expect(res.body).to.be.an('array');
  });
});

// Rebuild Instructions: Place in /vial/tests/integration/. Install dependencies: `npm install mocha chai supertest --save-dev`. Run `npx mocha tests/integration/server.test.js`. Check /vial/errorlog.md for issues.
