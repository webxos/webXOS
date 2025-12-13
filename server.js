// Local development server (optional)
const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// Import the crawler function
const { handler } = require('./api/crawler');

// Route for API endpoints
app.all('/api/*', async (req, res) => {
  // Convert Express req/res to serverless format
  const event = {
    path: req.path.replace('/api', ''),
    httpMethod: req.method,
    headers: req.headers,
    queryStringParameters: req.query,
    body: req.method === 'POST' ? JSON.stringify(req.body) : null,
    isBase64Encoded: false
  };
  
  const context = {};
  
  try {
    const result = await handler(event, context);
    
    // Set headers
    Object.entries(result.headers || {}).forEach(([key, value]) => {
      res.setHeader(key, value);
    });
    
    // Set status code
    res.status(result.statusCode || 200);
    
    // Send response
    if (result.isBase64Encoded) {
      res.send(Buffer.from(result.body, 'base64'));
    } else {
      res.send(result.body);
    }
  } catch (error) {
    console.error('Error in local server:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Serve the frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'crawl.html'));
});

app.get('/crawl.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'crawl.html'));
});

// Health check
app.get('/ping', (req, res) => {
  res.json({ status: 'ok', server: 'local' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Local server running on http://localhost:${PORT}`);
  console.log(`Frontend: http://localhost:${PORT}/crawl.html`);
  console.log(`API: http://localhost:${PORT}/api/crawl`);
  console.log(`Health check: http://localhost:${PORT}/ping`);
});