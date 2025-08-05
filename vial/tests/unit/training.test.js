const { trainVial } = require('../../src/tools/training');
const sqlite3 = require('sqlite3').verbose();

describe('Training', () => {
    let db;
    beforeAll(() => {
        db = new sqlite3.Database(':memory:');
        db.run(`
            CREATE TABLE vials (
                id TEXT PRIMARY KEY,
                code TEXT,
                training TEXT,
                status TEXT,
                latencyHistory TEXT,
                filePath TEXT,
                createdAt TEXT,
                codeLength INTEGER
            );
        `);
    });
    test('should train a vial', async () => {
        const req = { body: { id: 'test_vial', input: 'test' } };
        const res = { json: jest.fn(), status: jest.fn().mockReturnThis() };
        await trainVial(db, req, res);
        expect(res.json).toHaveBeenCalled();
    });
});

// Instructions:
# - Tests training
# - Run: `npm test`
