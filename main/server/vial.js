import React, { useState, useEffect } from 'https://cdn.jsdelivr.net/npm/react@18.2.0/+esm';
import ReactDOM from 'https://cdn.jsdelivr.net/npm/react-dom@18.2.0/+esm';
import axios from 'https://cdn.jsdelivr.net/npm/axios@1.7.7/+esm';
import 'https://cdn.jsdelivr.net/npm/tailwindcss@3.4.13/dist/tailwind.min.css';

const VialPage = () => {
  const [walletId, setWalletId] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [token, setToken] = useState('');
  const [notes, setNotes] = useState([]);
  const [content, setContent] = useState('');
  const [dbType, setDbType] = useState('postgres');
  const [message, setMessage] = useState('');

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

  const addNote = async () => {
    try {
      const response = await axios.post('https://localhost:8000/api/notes/add', {
        wallet_id: walletId,
        content,
        db_type: dbType
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessage(`Note added: ${response.data.note_id}`);
      setContent('');
    } catch (error) {
      setMessage(`Add note failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const fetchNotes = async () => {
    try {
      const response = await axios.post('https://localhost:8000/api/notes/read', {
        wallet_id: walletId,
        db_type: dbType,
        limit: 10
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setNotes(response.data.notes);
      setMessage('Notes fetched successfully');
    } catch (error) {
      setMessage(`Fetch notes failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  useEffect(() => {
    if (token) {
      fetchNotes();
    }
  }, [token]);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Vial MCP Controller</h1>
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
        <textarea
          placeholder="Note content"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="border p-2 w-full"
        />
        <select
          value={dbType}
          onChange={(e) => setDbType(e.target.value)}
          className="border p-2 mr-2"
        >
          <option value="postgres">PostgreSQL</option>
          <option value="mysql">MySQL</option>
          <option value="mongo">MongoDB</option>
        </select>
        <button onClick={addNote} className="bg-green-500 text-white p-2 rounded">
          Add Note
        </button>
      </div>
      <button onClick={fetchNotes} className="bg-purple-500 text-white p-2 rounded mb-4">
        Fetch Notes
      </button>
      <div>
        <h2 className="text-xl font-semibold">Notes</h2>
        <ul className="list-disc pl-5">
          {notes.map((note) => (
            <li key={note.id} className="my-2">
              {note.content} (ID: {note.id}, Timestamp: {note.timestamp})
            </li>
          ))}
        </ul>
      </div>
      {message && <p className="text-red-500">{message}</p>}
    </div>
  );
};

ReactDOM.render(<VialPage />, document.getElementById('root'));
