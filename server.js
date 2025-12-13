import express from 'express';
import cors from 'cors';
import * as cheerio from 'cheerio';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting - 100 requests per 15 minutes per IP
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});

// Apply rate limiting to crawl endpoint
app.use('/api/crawl', limiter);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// In-memory store for visited URLs (for demonstration only)
// In production, use Redis or database
const visitedCache = new Map();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

/**
 * Clean URL by removing fragments, trailing slashes, and normalizing
 */
function cleanUrl(url) {
  try {
    const urlObj = new URL(url);
    
    // Remove fragment
    urlObj.hash = '';
    
    // Remove trailing slash if not root
    if (urlObj.pathname.endsWith('/') && urlObj.pathname !== '/') {
      urlObj.pathname = urlObj.pathname.slice(0, -1);
    }
    
    // Normalize hostname to lowercase
    urlObj.hostname = urlObj.hostname.toLowerCase();
    
    // Remove default ports
    if (urlObj.port === '80' && urlObj.protocol === 'http:') {
      urlObj.port = '';
    } else if (urlObj.port === '443' && urlObj.protocol === 'https:') {
      urlObj.port = '';
    }
    
    return urlObj.toString();
  } catch {
    return url;
  }
}

/**
 * Normalize a URL (convert relative to absolute with proper base handling)
 */
function normalizeUrl(href, baseUrl) {
  try {
    // Trim whitespace
    href = href.trim();
    
    // Skip non-HTTP URLs
    if (href.startsWith('javascript:') || 
        href.startsWith('mailto:') || 
        href.startsWith('tel:') ||
        href.startsWith('#') ||
        href === '') {
      return null;
    }
    
    // Handle protocol-relative URLs (//example.com)
    if (href.startsWith('//')) {
      const base = new URL(baseUrl);
      href = base.protocol + href;
    }
    
    // Use the full baseUrl (not just origin) for proper path resolution
    const urlObj = new URL(href, baseUrl);
    
    // Clean the normalized URL
    return cleanUrl(urlObj.toString());
  } catch (error) {
    console.log('Failed to normalize URL:', href, error.message);
    return null;
  }
}

/**
 * Check if we should crawl this URL (simple robots.txt simulation)
 */
function shouldCrawl(url, startUrl) {
  try {
    const urlObj = new URL(url);
    const startUrlObj = new URL(startUrl);
    
    // For demo, only crawl same domain or subdomains
    if (!urlObj.hostname.endsWith(startUrlObj.hostname) && 
        !startUrlObj.hostname.endsWith(urlObj.hostname)) {
      return false;
    }
    
    return true;
  } catch {
    return false;
  }
}

// Health check endpoint
app.get('/api/ping', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'WEBXOS Crawler API is running',
    version: '2.0.0',
    timestamp: new Date().toISOString(),
    rateLimit: '100 requests per 15 minutes'
  });
});

// Stats endpoint
app.get('/api/stats', (req, res) => {
  res.json({
    visitedUrls: visitedCache.size,
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

// Main crawl endpoint
app.post('/api/crawl', async (req, res) => {
  const startTime = Date.now();
  
  try {
    const { startUrl } = req.body;
    
    if (!startUrl) {
      return res.status(400).json({ 
        error: 'Missing startUrl parameter',
        timestamp: new Date().toISOString()
      });
    }
    
    // Validate URL
    if (!startUrl.startsWith('http://') && !startUrl.startsWith('https://')) {
      return res.status(400).json({ 
        error: 'Invalid URL format. Must start with http:// or https://',
        timestamp: new Date().toISOString()
      });
    }
    
    // Check cache (with expiration)
    const cached = visitedCache.get(startUrl);
    if (cached && (Date.now() - cached.timestamp < CACHE_DURATION)) {
      return res.json({
        ...cached.data,
        cached: true,
        timestamp: new Date().toISOString()
      });
    }
    
    console.log(`[CRAWL] Starting: ${startUrl}`);
    
    // Set up timeout with AbortController
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout
    
    try {
      // Fetch with timeout and proper headers
      const response = await fetch(startUrl, {
        signal: controller.signal,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.5',
          'Accept-Encoding': 'gzip, deflate, br',
          'Connection': 'keep-alive',
          'Upgrade-Insecure-Requests': '1'
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      // Check content type
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('text/html')) {
        throw new Error('Not an HTML page');
      }
      
      const html = await response.text();
      const $ = cheerio.load(html);
      
      // Extract links with normalization and deduplication
      const linksSet = new Set();
      
      $('a[href]').each((i, element) => {
        const href = $(element).attr('href');
        if (href) {
          const normalized = normalizeUrl(href, startUrl);
          if (normalized && shouldCrawl(normalized, startUrl)) {
            linksSet.add(normalized);
          }
        }
      });
      
      // Limit to reasonable number
      const linksFound = Array.from(linksSet).slice(0, 50);
      
      // Extract metadata
      const title = $('title').text() || '';
      const description = $('meta[name="description"]').attr('content') || '';
      
      const result = {
        sourceUrl: startUrl,
        linksFound: linksFound,
        title: title,
        description: description,
        linkCount: linksFound.length,
        status: 'success',
        timestamp: new Date().toISOString(),
        duration: Date.now() - startTime
      };
      
      // Cache result
      visitedCache.set(startUrl, {
        data: result,
        timestamp: Date.now()
      });
      
      // Clean old cache entries
      if (visitedCache.size > 1000) {
        const keys = Array.from(visitedCache.keys());
        keys.slice(0, 500).forEach(key => visitedCache.delete(key));
      }
      
      console.log(`[CRAWL] Success: ${startUrl} (${linksFound.length} links)`);
      res.json(result);
      
    } catch (fetchError) {
      clearTimeout(timeoutId);
      throw fetchError;
    }
    
  } catch (error) {
    console.error(`[CRAWL] Error: ${error.message}`);
    
    let statusCode = 500;
    let errorMessage = 'Failed to crawl URL';
    
    if (error.name === 'AbortError') {
      statusCode = 408;
      errorMessage = 'Request timeout';
    } else if (error.message.includes('HTTP')) {
      statusCode = parseInt(error.message.match(/HTTP (\d+)/)?.[1]) || 500;
      errorMessage = error.message;
    } else if (error.message.includes('fetch failed')) {
      statusCode = 503;
      errorMessage = 'Network error';
    }
    
    res.status(statusCode).json({
      sourceUrl: req.body?.startUrl || 'unknown',
      linksFound: [],
      error: errorMessage,
      status: 'error',
      timestamp: new Date().toISOString(),
      duration: Date.now() - startTime
    });
  }
});

// Clear cache endpoint (for testing)
app.post('/api/clear-cache', (req, res) => {
  visitedCache.clear();
  res.json({
    message: 'Cache cleared',
    timestamp: new Date().toISOString()
  });
});

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/index.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('WEBXOS CRAWLER v2.0 - LOCAL DEVELOPMENT SERVER');
  console.log('='.repeat(60));
  console.log(`Server: http://localhost:${PORT}`);
  console.log(`API Health: http://localhost:${PORT}/api/ping`);
  console.log(`API Crawl: POST http://localhost:${PORT}/api/crawl`);
  console.log(`Rate Limit: 100 requests per 15 minutes`);
  console.log('='.repeat(60));
  console.log('Press Ctrl+C to stop the server');
});
