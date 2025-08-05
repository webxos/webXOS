const { createVial, getVials, destroyAllVials } = require('../../src/tools/vial_manager');
const sqlite3 = require('sqlite3').verbose();

describe('Vial Manager', () => {
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
    test('should create a vial', async () => {
        const req = { body: { id: 'test_vial', code: { js: 'console.log("Test");' }, training: { model: 'default', epochs: 5 } } };
        const res = { json: jest.fn(), status: jest.fn().mockReturnThis() };
        await createVial(db, require('ajv')(), req, res);
        expect(res.json).toHaveBeenCalled();
    });
});

// Instructions:
# - Tests vial creation
# - Run: `npm test`
