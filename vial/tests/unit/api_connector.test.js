const { fetchData } = require('../../src/tools/api_connector');

describe('API Connector', () => {
    test('should handle API error', async () => {
        const req = { body: { endpoint: 'https://invalid.api', method: 'GET' } };
        const res = { json: jest.fn(), status: jest.fn().mockReturnThis() };
        await fetchData(req, res);
        expect(res.status).toHaveBeenCalledWith(500);
    });
});

// Instructions:
# - Tests API connector
# - Run: `npm test`
