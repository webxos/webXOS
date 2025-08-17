import { Pool } from 'pg';
import { logger } from '../lib/logger';

export class DatabaseConfig {
  private pool: Pool;

  constructor() {
    const dbUrl = process.env.NEON_DATABASE_URL;
    if (!dbUrl) {
      throw new Error('NEON_DATABASE_URL not set');
    }
    this.pool = new Pool({
      connectionString: dbUrl,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000
    });
    this.pool.on('error', (err) => {
      logger.error(`Database pool error: ${err.message}`);
    });
  }

  async connect() {
    try {
      await this.pool.connect();
      logger.info('Database connection established');
    } catch (error) {
      logger.error(`Database connection error: ${error.message}`);
      throw error;
    }
  }

  async query(text: string, params: any[] = []) {
    try {
      const result = await this.pool.query(text, params);
      logger.info(`Query executed: ${text}`);
      return result;
    } catch (error) {
      logger.error(`Query error: ${error.message}`);
      throw error;
    }
  }

  async close() {
    try {
      await this.pool.end();
      logger.info('Database connection closed');
    } catch (error) {
      logger.error(`Database close error: ${error.message}`);
      throw error;
    }
  }
}
