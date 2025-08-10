```jsx
import React, { useState } from 'react';

function Auth({ isAuthenticated, onAuth, accessKey }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = () => {
    onAuth({ username, password });
  };

  const copyAccessKey = () => {
    navigator.clipboard.writeText(accessKey);
    alert('Access key copied to clipboard!');
  };

  return (
    <div className="auth">
      {!isAuthenticated ? (
        <>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button onClick={handleSubmit}>Authenticate</button>
        </>
      ) : (
        <div>
          <p>Access Key: {accessKey}</p>
          <button onClick={copyAccessKey}>Copy Access Key</button>
        </div>
      )}
    </div>
  );
}

export default Auth;
```
