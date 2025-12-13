const express = require('express');
const cors = require('cors');
const cheerio = require('cheerio');
const fetch = require('node-fetch');
const serverless = require('serverless-http');

// Initialize Express app
const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Store visited URLs in memory (for single function invocation)
// Note: For production, use Redis or database for persistence
const visitedUrls = new Set();

/**
 * Fetch and parse a URL to extract all links
 * @param {string} url - The URL to crawl
 * @returns {Promise<Array<string>>} Array of absolute URLs found on the page
 */
async function crawlUrl(url) {
  try {
    console.log(`Crawling: ${url}`);
    
    // Validate URL
    if (!isValidUrl(url)) {
      throw new Error(`Invalid URL: ${url}`);
    }

    // Fetch the page
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
      },
      timeout: 10000 // 10 second timeout
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
        href = normalizeUrl(href, url);
        
        // Filter out invalid URLs and non-HTTP protocols
        if (isValidUrl(href) && (href.startsWith('http://') || href.startsWith('https://'))) {
          // Remove fragments and query parameters for cleaner results
          const cleanUrl = removeFragmentAndQuery(href);
          links.add(cleanUrl);
        }
      }
    });

    // Also get links from other elements
    $('link[href], img[src], script[src], iframe[src]').each((i, element) => {
      const tagName = element.name;
      const attr = tagName === 'link' ? 'href' : 'src';
      let urlAttr = $(element).attr(attr);
      
      if (urlAttr) {
        urlAttr = normalizeUrl(urlAttr, url);
        if (isValidUrl(urlAttr)) {
          links.add(urlAttr);
        }
      }
    });

    return {
      sourceUrl: url,
      linksFound: Array.from(links).slice(0, 100), // Limit to 100 links per page
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
}

/**
 * Check if a string is a valid URL
 * @param {string} string - URL to validate
 * @returns {boolean} True if valid URL
 */
function isValidUrl(string) {
  try {
    const url = new URL(string);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch (_) {
    return false;
  }
}

/**
 * Normalize a URL (convert relative to absolute)
 * @param {string} href - URL to normalize
 * @param {string} baseUrl - Base URL for resolution
 * @returns {string} Normalized absolute URL
 */
function normalizeUrl(href, baseUrl) {
  try {
    // Remove whitespace and trim
    href = href.trim();
    
    // Skip javascript:, mailto:, tel:, etc.
    if (href.startsWith('javascript:') || 
        href.startsWith('mailto:') || 
        href.startsWith('tel:') ||
        href.startsWith('#')) {
      return href;
    }
    
    // Handle protocol-relative URLs
    if (href.startsWith('//')) {
      href = 'https:' + href;
    }
    
    // Handle relative URLs
    if (href.startsWith('/')) {
      const base = new URL(baseUrl);
      href = base.origin + href;
    } else if (!href.startsWith('http')) {
      const base = new URL(baseUrl);
      href = new URL(href, base.origin).href;
    }
    
    return href;
  } catch (_) {
    return href;
  }
}

/**
 * Remove fragment and query parameters from URL
 * @param {string} url - URL to clean
 * @returns {string} Cleaned URL
 */
function removeFragmentAndQuery(url) {
  try {
    const urlObj = new URL(url);
    urlObj.hash = '';
    urlObj.search = '';
    return urlObj.href;
  } catch (_) {
    return url;
  }
}

// Routes

// Health check endpoint
app.get('/ping', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'WEBXOS Crawler API is running',
    version: '2.0.0',
    timestamp: new Date().toISOString()
  });
});

// Single URL crawl endpoint
app.post('/crawl', async (req, res) => {
  try {
    const { startUrl } = req.body;
    
    if (!startUrl) {
      return res.status(400).json({ 
        error: 'Missing startUrl parameter',
        timestamp: new Date().toISOString()
      });
    }
    
    if (!isValidUrl(startUrl)) {
      return res.status(400).json({ 
        error: 'Invalid URL format',
        timestamp: new Date().toISOString()
      });
    }
    
    // Add rate limiting check (basic)
    if (visitedUrls.has(startUrl)) {
      return res.status(429).json({
        sourceUrl: startUrl,
        linksFound: [],
        error: 'URL already visited recently',
        timestamp: new Date().toISOString(),
        status: 'rate_limited'
      });
    }
    
    // Add to visited (with expiration - simple approach)
    visitedUrls.add(startUrl);
    
    // Clean up visited URLs if too many (prevent memory issues)
    if (visitedUrls.size > 1000) {
      // Convert to array, remove first 500
      const urlsArray = Array.from(visitedUrls);
      visitedUrls.clear();
      urlsArray.slice(500).forEach(url => visitedUrls.add(url));
    }
    
    const result = await crawlUrl(startUrl);
    
    // Add CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
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

// Batch crawl endpoint (optional, for future enhancement)
app.post('/crawl/batch', async (req, res) => {
  try {
    const { urls, maxConcurrent = 3 } = req.body;
    
    if (!urls || !Array.isArray(urls)) {
      return res.status(400).json({ 
        error: 'Missing or invalid urls array',
        timestamp: new Date().toISOString()
      });
    }
    
    // Limit number of URLs
    const limitedUrls = urls.slice(0, 10);
    
    // Process URLs concurrently with limit
    const results = [];
    for (let i = 0; i < limitedUrls.length; i += maxConcurrent) {
      const batch = limitedUrls.slice(i, i + maxConcurrent);
      const batchPromises = batch.map(url => crawlUrl(url));
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
      
      // Add delay between batches
      if (i + maxConcurrent < limitedUrls.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    res.json({
      results,
      totalProcessed: results.length,
      timestamp: new Date().toISOString(),
      status: 'completed'
    });
  } catch (error) {
    console.error('Batch crawl error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get crawler stats
app.get('/stats', (req, res) => {
  res.json({
    visitedUrls: visitedUrls.size,
    uptime: process.uptime(),
    memoryUsage: process.memoryUsage(),
    timestamp: new Date().toISOString()
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'WEBXOS Crawler API',
    version: '2.0.0',
    endpoints: {
      'GET /': 'API information',
      'GET /ping': 'Health check',
      'POST /crawl': 'Crawl single URL',
      'POST /crawl/batch': 'Crawl multiple URLs',
      'GET /stats': 'Crawler statistics'
    },
    documentation: 'See frontend at /index.html or /crawl.html'
  });
});

// Handle 404
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.path,
    timestamp: new Date().toISOString()
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined,
    timestamp: new Date().toISOString()
  });
});

// Export for Netlify Functions
module.exports.handler = serverless(app);

// For local development
if (require.main === module) {
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`WEBXOS Crawler API running on http://localhost:${PORT}`);
    console.log(`- Health check: http://localhost:${PORT}/ping`);
    console.log(`- API info: http://localhost:${PORT}/`);
  });
}