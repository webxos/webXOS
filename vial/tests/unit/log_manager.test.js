const { syncLog } = require('../../src/tools/log_manager');
const sqlite3 = require('sqlite3').verbose();
const LZString = require('lz-string');

describe('Log Manager', () => {
    let db;
    beforeAll(() => {
        db = new sqlite3.Database(':memory:');
        db.run(`
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                message TEXT,
                metadata TEXT,
                urgency TEXT
            );
        `);
    });
    test('should sync a log', async () => {
        const log = LZString.compressToUTF16(JSON.stringify({ timestamp: new Date().toISOString(), event_type: 'test', message: 'Test log', metadata: {}, urgency: 'LOW' }));
        const req = { body: { log } };
        const res = { json: jest.fn(), status: jest.fn().mockReturnThis() };
        await syncLog(db, LZString, req, res);
        expect(res.json).toHaveBeenCalledWith({ message: 'Log synced' });
    });
});

// Instructions:
# - Tests log sync
# - Run: `npm test`
