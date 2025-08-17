const { Pool } = require('@neondatabase/serverless');
const fs = require('fs').promises;

async function migrate() {
    const pool = new Pool({ connectionString: process.env.NEON_DATABASE_URL });
    try {
        const client = await pool.connect();
        const schema = await fs.readFile('database/schema.sql', 'utf-8');
        await client.query(schema);
        console.log('Database migration completed');
        client.release();
        await pool.end();
    } catch (err) {
        console.error('Migration failed:', err.message);
        process.exit(1);
    }
}

migrate();
