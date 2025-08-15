import React, { Suspense, lazy, useState } from 'react';
import { Container, Navbar, Alert } from 'react-bootstrap';
import './App.css';
import { initiateOAuth, handleCallback } from './utils/oauth';

const Console = lazy(() => import('./components/Console'));
const ButtonGroup = lazy(() => import('./components/ButtonGroup'));

function App() {
  const [logs, setLogs] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState(null);

  const log = (message, isCommand = false) => {
    setLogs((prev) => {
      const newLogs = [...prev, `[${new Date().toLocaleTimeString()}] ${isCommand ? `<strong>${message}</strong>` : message}`];
      return newLogs.slice(-50);
    });
  };

  const handleTroubleshoot = async () => {
    log('Running system diagnostics...', true);
    // Simulate diagnostics
    setTimeout(() => log('Troubleshoot complete: System OK', true), 1000);
  };

  const handleOAuth = async () => {
    log('Initiating OAuth authentication...', true);
    try {
      const redirectUrl = await initiateOAuth();
      if (redirectUrl) {
        window.location.href = redirectUrl;
      } else {
        setIsAuthenticated(true);
        log('OAuth authentication successful', true);
      }
    } catch (err) {
      log(`OAuth error: ${err.message}`, true);
      setError(err.message);
    }
  };

  const handleDashboard = async () => {
    if (!isAuthenticated) {
      log('Please authenticate via OAuth first', true);
      return;
    }
    log('Accessing dashboard...', true);
    // Simulate dashboard access
    setTimeout(() => log('Dashboard access granted', true), 1000);
  };

  // Handle OAuth callback
  React.useEffect(() => {
    if (window.location.search.includes('code=')) {
      handleCallback()
        .then(() => {
          setIsAuthenticated(true);
          log('OAuth authentication successful', true);
          window.history.replaceState({}, document.title, window.location.pathname);
        })
        .catch((err) => {
          log(`OAuth callback error: ${err.message}`, true);
          setError(err.message);
        });
    }
  }, []);

  return (
    <div className="App">
      <Navbar bg="dark" variant="dark">
        <Container>
          <Navbar.Brand>Vial MCP Gateway</Navbar.Brand>
        </Container>
      </Navbar>
      <Container className="mt-3">
        {error && <Alert variant="danger">{error}</Alert>}
        <Suspense fallback={<div>Loading...</div>}>
          <Console logs={logs} />
          <ButtonGroup
            onTroubleshoot={handleTroubleshoot}
            onOAuth={handleOAuth}
            onDashboard={handleDashboard}
            isAuthenticated={isAuthenticated}
          />
        </Suspense>
        <footer className="text-center text-success mt-3">
          &copy; 2025 WebXOS - Vial MCP Gateway<br />
          Troubleshoot: Run system diagnostics | OAuth: Authenticate | Dashboard: Access after authentication
        </footer>
      </Container>
    </div>
  );
}

export default App;
