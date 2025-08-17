const { Pool } = require('pg');
const logger = require('./logger');

class NeonDB {
  constructor() {
    this.pool = new Pool({
      connectionString: process.env.NEON_DATABASE_URL,
      ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
    });
    this.pool.on('error', (err) => {
      logger.error(`Database error: ${err.message}`);
    });
  }

  async query(text, params) {
    try {
      const start = Date.now();
      const res = await this.pool.query(text, params);
      logger.info(`Query executed: ${text} in ${Date.now() - start}ms`);
      return res;
    } catch (err) {
      logger.error(`Query error: ${err.message} for query: ${text}`);
      throw err;
    }
  }

  async connect() {
    try {
      const client = await this.pool.connect();
      logger.info('Database connected');
      client.release();
    } catch (err) {
      logger.error(`Database connection error: ${err.message}`);
      throw err;
    }
  }

  async close() {
    try {
      await this.pool.end();
      logger.info('Database connection closed');
    } catch (err) {
      logger.error(`Database close error: ${err.message}`);
      throw err;
    }
  }
}

module.exports = { NeonDB };
