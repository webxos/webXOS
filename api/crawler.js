import * as cheerio from 'cheerio';

// Simple in-memory cache for rate limiting (per function instance)
const requestCounts = new Map();
const RATE_LIMIT_WINDOW = 60 * 1000; // 1 minute
const RATE_LIMIT_MAX = 30; // 30 requests per minute per IP

/**
 * Check rate limit
 */
function checkRateLimit(ip) {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;
  
  let requests = requestCounts.get(ip) || [];
  
  // Remove old requests
  requests = requests.filter(time => time > windowStart);
  
  if (requests.length >= RATE_LIMIT_MAX) {
    return false;
  }
  
  requests.push(now);
  requestCounts.set(ip, requests);
  return true;
}

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
    console.log('Failed to normalize URL:', href);
    return null;
  }
}

export const handler = async (event, context) => {
  // Set CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Content-Type': 'application/json'
  };

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  try {
    // Extract client IP for rate limiting
    const clientIp = event.headers['x-nf-client-connection-ip'] || 
                    event.headers['x-forwarded-for'] || 
                    'unknown';
    
    // Apply rate limiting
    if (!checkRateLimit(clientIp)) {
      return {
        statusCode: 429,
        headers,
        body: JSON.stringify({
          error: 'Rate limit exceeded. Please try again later.',
          timestamp: new Date().toISOString()
        })
      };
    }

    // Handle different endpoints
    const path = event.path.replace('/.netlify/functions/crawler', '');
    
    // Ping endpoint
    if (path === '/ping' && event.httpMethod === 'GET') {
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
          status: 'ok',
          message: 'WEBXOS Crawler API is running',
          version: '2.0.0',
          timestamp: new Date().toISOString(),
          rateLimit: '30 requests per minute'
        })
      };
    }
    
    // Crawl endpoint
    if (path === '/crawl' && event.httpMethod === 'POST') {
      const body = JSON.parse(event.body || '{}');
      const { startUrl } = body;
      
      if (!startUrl) {
        return {
          statusCode: 400,
          headers,
          body: JSON.stringify({ 
            error: 'Missing startUrl parameter',
            timestamp: new Date().toISOString()
          })
        };
      }
      
      if (!startUrl.startsWith('http://') && !startUrl.startsWith('https://')) {
        return {
          statusCode: 400,
          headers,
          body: JSON.stringify({ 
            error: 'Invalid URL format. Must start with http:// or https://',
            timestamp: new Date().toISOString()
          })
        };
      }
      
      console.log(`[NETLIFY] Crawling: ${startUrl}`);
      
      // Set up timeout with AbortController (Netlify functions have 10s timeout)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);
      
      try {
        // Fetch with timeout
        const response = await fetch(startUrl, {
          signal: controller.signal,
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
          }
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          return {
            statusCode: response.status,
            headers,
            body: JSON.stringify({
              error: `HTTP ${response.status}: ${response.statusText}`,
              timestamp: new Date().toISOString()
            })
          };
        }
        
        const html = await response.text();
        const $ = cheerio.load(html);
        
        // Extract links with proper normalization
        const linksSet = new Set();
        
        $('a[href]').each((i, element) => {
          const href = $(element).attr('href');
          if (href) {
            const normalized = normalizeUrl(href, startUrl);
            if (normalized) {
              linksSet.add(normalized);
            }
          }
        });
        
        // Extract metadata
        const title = $('title').text() || '';
        
        const result = {
          sourceUrl: startUrl,
          linksFound: Array.from(linksSet).slice(0, 30), // Limit for Netlify
          title: title,
          status: 'success',
          timestamp: new Date().toISOString()
        };
        
        console.log(`[NETLIFY] Success: ${startUrl} (${result.linksFound.length} links)`);
        
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify(result)
        };
        
      } catch (fetchError) {
        clearTimeout(timeoutId);
        throw fetchError;
      }
    }
    
    // 404 for unknown endpoints
    return {
      statusCode: 404,
      headers,
      body: JSON.stringify({
        error: 'Endpoint not found',
        path: path,
        timestamp: new Date().toISOString()
      })
    };
    
  } catch (error) {
    console.error('[NETLIFY] Error:', error.message);
    
    let statusCode = 500;
    let errorMessage = 'Internal server error';
    
    if (error.name === 'AbortError') {
      statusCode = 408;
      errorMessage = 'Request timeout';
    } else if (error.message.includes('HTTP')) {
      errorMessage = error.message;
    }
    
    return {
      statusCode: statusCode,
      headers,
      body: JSON.stringify({
        error: errorMessage,
        timestamp: new Date().toISOString()
      })
    };
  }
};
