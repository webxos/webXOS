const request = require('supertest');
const app = require('../../src/server');

describe('Server Integration', () => {
    test('should serve vial.html', async () => {
        const res = await request(app).get('/');
        expect(res.status).toBe(200);
        expect(res.text).toContain('Vial MCP Controller');
    });
    test('should list tools', async () => {
        const res = await request(app).get('/mcp/tools');
        expect(res.status).toBe(200);
        expect(res.body.tools).toContain('vial_manager');
    });
});

// Instructions:
# - Tests server endpoints
# - Run: `npm test`
