const express = require('express');
const { performSearch, loadSiteIndex } = require('../utils/search');
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(express.json());
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
});

let siteIndex = [];

async function initialize() {
    siteIndex = await loadSiteIndex(fs, path, '../utils/site_index.json');
}

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', version: '1.0.0' });
});

app.post('/search', async (req, res) => {
    const { query } = req.body;
    if (!query) {
        return res.status(400).json({ error: 'Missing query' });
    }
    const results = await performSearch(query, siteIndex, 'server-agent1');
    res.json({ results });
});

initialize();

module.exports.handler = async (event, context) => {
    const { httpMethod, path, body } = event;
    if (httpMethod === 'GET' && path === '/health') {
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
            body: JSON.stringify({ status: 'healthy', version: '1.0.0' })
        };
    } else if (httpMethod === 'POST' && path === '/search') {
        const { query } = JSON.parse(body || '{}');
        if (!query) {
            return {
                statusCode: 400,
                headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                body: JSON.stringify({ error: 'Missing query' })
            };
        }
        const results = await performSearch(query, siteIndex, 'server-agent1');
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
            body: JSON.stringify({ results })
        };
    }
    return {
        statusCode: 405,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({ error: 'Method not allowed' })
    };
};
