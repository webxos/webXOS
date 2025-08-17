import { DatabaseConfig } from '../config/database-config';
import { logger } from '../lib/logger';
import { OAuth2Client } from 'google-auth-library';
import * as jwt from 'jsonwebtoken';

interface AuthInput {
  oauth_token: string;
  provider: 'google';
}

interface AuthOutput {
  access_token: string;
  expires_in: number;
  user_id: string;
}

export class AuthenticationTool {
  private db: DatabaseConfig;
  private oauthClient: OAuth2Client;

  constructor(db: DatabaseConfig) {
    this.db = db;
    const clientId = process.env.GOOGLE_CLIENT_ID;
    if (!clientId) {
      throw new Error('GOOGLE_CLIENT_ID not set');
    }
    this.oauthClient = new OAuth2Client(clientId);
  }

  async execute(input: AuthInput): Promise<AuthOutput> {
    try {
      if (input.provider !== 'google') {
        throw new Error('Unsupported OAuth provider');
      }

      const ticket = await this.oauthClient.verifyIdToken({
        idToken: input.oauth_token,
        audience: process.env.GOOGLE_CLIENT_ID
      });

      const payload = ticket.getPayload();
      if (!payload?.sub || !payload?.email) {
        throw new Error('Invalid OAuth token');
      }

      const user_id = `user_${payload.sub}`;
      const email = payload.email;
      const username = payload.name || email.split('@')[0];

      const user = await this.db.query('SELECT user_id FROM users WHERE user_id = $1', [user_id]);
      if (!user.rows.length) {
        await this.db.query(
          'INSERT INTO users (user_id, username, wallet_address, balance, reputation, created_at) VALUES ($1, $2, $3, $4, $5, $6)',
          [user_id, username, `wallet_${user_id}`, 0, 0, new Date()]
        );
      }

      const access_token = jwt.sign({ user_id }, process.env.JWT_SECRET || 'default_secret', { expiresIn: '24h' });
      logger.info(`User authenticated: ${user_id} via ${input.provider}`);
      return {
        access_token,
        expires_in: 86400,
        user_id
      };
    } catch (error) {
      logger.error(`Authentication error: ${error.message}`);
      throw error;
    }
  }
}
