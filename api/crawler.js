// /api/crawler.js

// Import necessary libraries
const express = require('express');
const fetch = require('node-fetch');
const cheerio = require('cheerio');
const cors = require('cors'); // Required to allow your frontend to access this API
const serverless = require('serverless-http');

const app = express();
app.use(express.json());
app.use(cors()); // Enable CORS for all routes

// A simple in-memory structure to prevent immediate re-crawling within one session, 
// though a production crawler would need a database like Redis/MongoDB.
const visitedUrls = new Set();
const crawlingQueue = [];

// API Endpoint to start the crawl
app.post('/crawl', async (req, res) => {
    const { startUrl } = req.body;

    if (!startUrl || visitedUrls.has(startUrl)) {
        return res.status(400).json({ error: 'Invalid or already visited URL' });
    }

    crawlingQueue.push(startUrl);
    visitedUrls.add(startUrl);

    // In a simple API call, we just process the initial URL and return its links.
    // A true long-running crawler would process the entire queue in the background.
    try {
        const links = await fetchAndParseLinks(startUrl);
        res.json({ 
            sourceUrl: startUrl, 
            linksFound: links,
            message: `Found ${links.length} links on the page.`
        });
    } catch (error) {
        res.status(500).json({ error: 'Failed to crawl the target URL' });
    }
});

/**
 * Fetches an HTML page, parses it using Cheerio, and extracts all links.
 * @param {string} url 
 * @returns {Promise<string[]>} List of absolute URLs found on the page.
 */
async function fetchAndParseLinks(url) {
    console.log(`Fetching: ${url}`);
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const html = await response.text();
    const $ = cheerio.load(html);
    const links = [];
    const baseUrl = new URL(url);

    $('a').each((index, element) => {
        const relativeUrl = $(element).attr('href');
        if (relativeUrl) {
            try {
                // Convert relative URLs to absolute URLs
                const absoluteUrl = new URL(relativeUrl, baseUrl).href;
                if (!visitedUrls.has(absoluteUrl)) {
                    // You can add logic here to filter domains if needed
                    links.push(absoluteUrl);
                    // visitedUrls.add(absoluteUrl); // Mark as visited immediately if processing recursively
                }
            } catch (e) {
                // Ignore invalid URLs
            }
        }
    });
    return links;
}

// Export for serverless
module.exports = app;
module.exports.handler = serverless(app);
