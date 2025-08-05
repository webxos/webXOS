// src/tools/api_connector.js
const redaxios = require('redaxios');

exports.fetchData = async (req, res) => {
    try {
        const { endpoint, method = 'GET', headers = {}, body } = req.body;
        const response = await redaxios({ url: endpoint, method, headers, data: body });
        res.json(response.data);
    } catch (err) {
        console.error(`[API_CONNECTOR] Error: ${err.message}`);
        res.status(500).json({ error: err.message });
    }
};

// Instructions:
// - Fetches external API data
// - Uses redaxios for lightweight HTTP
