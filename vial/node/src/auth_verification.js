const axios = require('axios');

const authVerification = {
  async verifyOAuthToken(code) {
    try {
      const response = await axios.post(
        'https://github.com/login/oauth/access_token',
        {
          client_id: process.env.OAUTH_CLIENT_ID,
          client_secret: process.env.OAUTH_CLIENT_SECRET,
          code
        },
        { headers: { Accept: 'application/json' } }
      );

      if (response.data.error) {
        throw new Error(response.data.error_description);
      }

      const userResponse = await axios.get('https://api.github.com/user', {
        headers: { Authorization: `Bearer ${response.data.access_token}` }
      });

      return {
        user_id: userResponse.data.login,
        roles: ['read:data', 'read:llm', 'write:git'],
        access_token: response.data.access_token
      };
    } catch (error) {
      console.error(`OAuth verification error: ${error.message}`);
      throw new Error(`OAuth verification failed: ${error.message}`);
    }
  }
};

module.exports = authVerification;
