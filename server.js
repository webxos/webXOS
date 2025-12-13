// Local development server
const express = require('express');
const path = require('path');
const cors = require('cors');
const cheerio = require('cheerio');
const fetch = require('node-fetch');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// Import the crawler function directly (not as Netlify function for local dev)
const crawlUrl = async (url) => {
  try {
    console.log(`Crawling: ${url}`);
    
    // Validate URL
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      throw new Error(`Invalid URL: ${url}`);
    }

    // Fetch the page
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      },
      timeout: 10000
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const html = await response.text();
    const $ = cheerio.load(html);
    
    // Extract all links
    const links = new Set();
    
    // Get all anchor tags
    $('a[href]').each((i, element) => {
      let href = $(element).attr('href');
      
      if (href) {
        // Convert relative URLs to absolute
        if (href.startsWith('/')) {
          const base = new URL(url);
          href = base.origin + href;
        } else if (href.startsWith('http')) {
          // Already absolute
        } else {
          // Relative path
          const base = new URL(url);
          href = new URL(href, base.origin).href;
        }
        
        // Filter out non-HTTP protocols
        if (href.startsWith('http://') || href.startsWith('https://')) {
          links.add(href);
        }
      }
    });

    return {
      sourceUrl: url,
      linksFound: Array.from(links).slice(0, 50),
      title: $('title').text() || '',
      timestamp: new Date().toISOString(),
      status: 'success'
    };
  } catch (error) {
    console.error(`Error crawling ${url}:`, error.message);
    return {
      sourceUrl: url,
      linksFound: [],
      error: error.message,
      timestamp: new Date().toISOString(),
      status: 'error'
    };
  }
};

// Health check endpoint
app.get('/api/ping', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'WEBXOS Crawler API is running (local)',
    version: '2.0.0',
    timestamp: new Date().toISOString()
  });
});

// Crawl endpoint
app.post('/api/crawl', async (req, res) => {
  try {
    const { startUrl } = req.body;
    
    if (!startUrl) {
      return res.status(400).json({ 
        error: 'Missing startUrl parameter',
        timestamp: new Date().toISOString()
      });
    }
    
    const result = await crawlUrl(startUrl);
    
    res.json(result);
  } catch (error) {
    console.error('Crawl endpoint error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Options for CORS preflight
app.options('/api/crawl', (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.status(200).end();
});

// Serve the frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'crawl.html'));
});

app.get('/crawl.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'crawl.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`=========================================`);
  console.log(`WEBXOS Crawler v2.0 - Local Development`);
  console.log(`=========================================`);
  console.log(`Frontend: http://localhost:${PORT}/crawl.html`);
  console.log(`API Health: http://localhost:${PORT}/api/ping`);
  console.log(`API Crawl: POST http://localhost:${PORT}/api/crawl`);
  console.log(`=========================================`);
  console.log(`Press Ctrl+C to stop the server`);
});
