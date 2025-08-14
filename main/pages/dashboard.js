# server/pages/dashboard.js
import React, { useState, useEffect } from 'https://cdn.jsdelivr.net/npm/react@18.2.0/+esm';
import ReactDOM from 'https://cdn.jsdelivr.net/npm/react-dom@18.2.0/+esm';
import 'https://cdn.tailwindcss.com/3.3.3';

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        setError('No token found. Please login.');
        return;
      }
      try {
        const response = await fetch('/api/auth/users/me', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        if (response.ok) {
          setUser(data);
        } else {
          setError(data.detail);
        }
      } catch (err) {
        setError('Failed to fetch user data');
      }
    };
    fetchUser();
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <h1 className="text-3xl font-bold mb-4">Vial MCP Dashboard</h1>
      {error && <p className="text-red-500">{error}</p>}
      {user && (
        <div className="bg-white p-6 rounded shadow-md">
          <h2 className="text-xl font-semibold">Welcome, {user.username}!</h2>
          <p className="mt-2">Manage your agents and tasks here.</p>
        </div>
      )}
    </div>
  );
};

ReactDOM.render(<Dashboard />, document.getElementById('root'));
