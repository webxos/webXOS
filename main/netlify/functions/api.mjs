import { serve } from 'serverless-http';
import { createApp } from './main.py'; // Assuming main.py is bundled as ESM
import { v4 as uuidv4 } from 'uuid';

const app = createApp();
const handler = serve(app);

// Generate CSP nonce
const generateNonce = () => {
  return Buffer.from(uuidv4()).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
};

export default async (req, context) => {
  const nonce = generateNonce();
  
  // Inject nonce into HTML responses
  if (req.path === '/' || req.path === '/index.html') {
    const response = await handler(req, context);
    if (response.statusCode === 200 && response.headers['content-type'].includes('text/html')) {
      const body = Buffer.from(response.body, 'base64').toString('utf-8');
      const modifiedBody = body.replace(/{{nonce}}/g, nonce);
      return {
        ...response,
        body: Buffer.from(modifiedBody).toString('base64'),
        headers: {
          ...response.headers,
          'Content-Security-Policy': response.headers['Content-Security-Policy'].replace('{{nonce}}', nonce)
        }
      };
    }
    return response;
  }

  return handler(req, context);
};
