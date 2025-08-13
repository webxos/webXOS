import axios from 'axios';

const cognitallmwareAgent = {
  async search(query, userId, apiKey) {
    try {
      const response = await axios.post(
        '/v1/api/cognitallmware_search',
        { user_id: userId, query, limit: 5 },
        { headers: { Authorization: `Bearer ${apiKey}` } }
      );
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      console.error(`CogniTALLMware search error: ${errorMessage}`);
      throw new Error(`CogniTALLMware search failed: ${errorMessage}`);
    }
  },

  async displayResults(results, containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error('Results container not found');
      return;
    }
    container.innerHTML = '';
    results.data.matches.forEach(match => {
      const div = document.createElement('div');
      div.textContent = `${match.id}: ${match.data} (Score: ${match.score})`;
      container.appendChild(div);
    });
  }
};

export default cognitallmwareAgent;
