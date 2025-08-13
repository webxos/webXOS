import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const response = await axios.post('http://unified_server:8000/v1/api/retrieve', req.body, {
      headers: { Authorization: req.headers.authorization }
    });
    res.status(200).json(response.data);
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.message;
    console.error(`Retrieve API error: ${errorMessage}`);
    res.status(error.response?.status || 500).json({ error: `Retrieval failed: ${errorMessage}` });
  }
}
