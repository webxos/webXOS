const axios = require('axios');

const networkSync = {
  async syncAgents(userId, agents) {
    try {
      const endpoints = {
        nomic: '/v1/api/nomic_search',
        cognitallmware: '/v1/api/cognitallmware_search',
        llmware: '/v1/api/llmware_search',
        jinaai: '/v1/api/jinaai_search'
      };

      const results = {};
      for (const agent of agents) {
        if (!endpoints[agent]) {
          throw new Error(`Unknown agent: ${agent}`);
        }
        const response = await axios.post(
          `http://unified_server:8000${endpoints[agent]}`,
          { user_id: userId, query: 'sync_check', limit: 1 },
          { headers: { Authorization: `Bearer ${process.env.API_KEY}` } }
        );
        results[agent] = response.data;
      }

      console.log(`Network sync completed for user ${userId}`);
      return { status: 'success', data: results };
    } catch (error) {
      console.error(`Network sync error: ${error.message}`);
      return { status: 'error', error: error.message };
    }
  }
};

module.exports = networkSync;
