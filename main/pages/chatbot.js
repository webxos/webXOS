import React, { useState, useEffect } from 'https://cdn.jsdelivr.net/npm/react@18.2.0/+esm';
import ReactDOM from 'https://cdn.jsdelivr.net/npm/react-dom@18.2.0/+esm';
import axios from 'https://cdn.jsdelivr.net/npm/axios@1.7.7/+esm';
import 'https://cdn.jsdelivr.net/npm/tailwindcss@3.4.13/dist/tailwind.min.css';

const ChatbotPage = () => {
  const [walletId, setWalletId] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [token, setToken] = useState('');
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [inputText, setInputText] = useState('');
  const [agentName, setAgentName] = useState('jina');
  const [targetLang, setTargetLang] = useState('es');

  const login = async () => {
    try {
      const response = await axios.post('https://localhost:8000/api/auth/login', {
        wallet_id: walletId,
        api_key: apiKey
      });
      setToken(response.data.access_token);
      setMessage('Login successful');
    } catch (error) {
      setMessage(`Login failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;
    try {
      const response = await axios.post('https://localhost:8000/api/translate', {
        wallet_id: walletId,
        content: inputText,
        target_lang: targetLang
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const newMessage = {
        user: inputText,
        bot: response.data.translated_content,
        timestamp: new Date().toISOString()
      };
      setChatHistory([...chatHistory, newMessage]);
      setInputText('');
      setMessage('Message sent successfully');
    } catch (error) {
      setMessage(`Message failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  useEffect(() => {
    if (token) {
      setChatHistory([]);
    }
  }, [token]);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Vial MCP Chatbot</h1>
      <div className="mb-4">
        <input
          type="text"
          placeholder="Wallet ID"
          value={walletId}
          onChange={(e) => setWalletId(e.target.value)}
          className="border p-2 mr-2"
        />
        <input
          type="text"
          placeholder="API Key"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          className="border p-2 mr-2"
        />
        <button onClick={login} className="bg-blue-500 text-white p-2 rounded">
          Login
        </button>
      </div>
      <div className="mb-4">
        <select
          value={agentName}
          onChange={(e) => setAgentName(e.target.value)}
          className="border p-2 mr-2"
        >
          <option value="jina">Jina AI (Translation)</option>
          <option value="nomic">Nomic</option>
          <option value="cognitallmware">CogniTALLMware</option>
          <option value="llmware">LLMware</option>
        </select>
        <select
          value={targetLang}
          onChange={(e) => setTargetLang(e.target.value)}
          className="border p-2 mr-2"
        >
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
        </select>
      </div>
      <div className="mb-4">
        <textarea
          placeholder="Type your message..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="border p-2 w-full"
        />
        <button onClick={sendMessage} className="bg-green-500 text-white p-2 rounded mt-2">
          Send Message
        </button>
      </div>
      <div>
        <h2 className="text-xl font-semibold">Chat History</h2>
        <div className="border p-4 h-64 overflow-y-auto">
          {chatHistory.map((msg, index) => (
            <div key={index} className="mb-2">
              <p><strong>User:</strong> {msg.user}</p>
              <p><strong>Bot:</strong> {msg.bot} <span className="text-gray-500">({msg.timestamp})</span></p>
            </div>
          ))}
        </div>
      </div>
      {message && <p className="text-red-500">{message}</p>}
    </div>
  );
};

ReactDOM.render(<ChatbotPage />, document.getElementById('root'));
