```jsx
import React, { useState, useRef, useEffect } from 'react';

function Console({ logs, isAuthenticated, onPrompt }) {
  const [prompt, setPrompt] = useState('');
  const consoleRef = useRef(null);

  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [logs]);

  const handleSubmit = (e) => {
    if (e.key === 'Enter' && prompt.trim()) {
      onPrompt(prompt);
      setPrompt('');
    }
  };

  return (
    <div id="console" className={isAuthenticated ? 'active-monitor' : ''}>
      {logs.map((log, index) => (
        <p key={index} className={log.includes('error') ? 'error' : 'command'}>{log}</p>
      ))}
      {isAuthenticated && (
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyPress={handleSubmit}
          placeholder="Enter prompt for Communications Agent..."
          className="prompt-input"
        />
      )}
    </div>
  );
}

export default Console;
```
