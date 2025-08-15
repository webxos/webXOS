import React, { Suspense, lazy, useState, Component } from 'react';
import { Container, Navbar, Alert } from 'react-bootstrap';
import './App.css';
import { initiateOAuth, handleCallback } from './utils/oauth';

const Console = lazy(() => import('./components/Console'));
const ButtonGroup = lazy(() => import('./components/ButtonGroup'));

class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error: error.message };
  }

  render() {
    if (this.state.error) {
      return (
        <Container className="mt-3">
          <Alert variant="danger">Error: {this.state.error}. Please refresh or check the console.</Alert>
        </Container>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [logs, setLogs] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('oauth_token'));
  const [error, setError] = useState(null);

  const log = (message, isCommand = false) => {
    setLogs((prev) => {
      const newLogs = [...prev, `[${new Date().toLocaleTimeString()}] ${isCommand ? `<strong>${message}</strong>` : message}`];
      return newLogs.slice(-50);
    });
  };

  const handleTroubleshoot = async () => {
    log('Running system diagnostics...', true);
    try {
      // Simulate diagnostics
      setTimeout(() => log('Troubleshoot complete: System OK', true), 1000);
    } catch (err) {
      log(`Troubleshoot error: ${err.message}`, true);
      setError(err.message);
    }
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
    try {
      // Simulate dashboard access
      setTimeout(() => log('Dashboard access granted', true), 1000);
    } catch (err) {
      log(`Dashboard error: ${err.message}`, true);
      setError(err.message);
    }
  };

  React.useEffect(() => {
    log('Vial MCP Gateway initialized', true);
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
    <ErrorBoundary>
      <div className="App">
        <Navbar bg="dark" variant="dark">
          <Container>
            <Navbar.Brand>Vial MCP Gateway</Navbar.Brand>
          </Container>
        </Navbar>
        <Container className="mt-3">
          {error && <Alert variant="danger">{error}</Alert>}
          <Suspense fallback={<div className="text-success">Loading...</div>}>
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
    </ErrorBoundary>
  );
}

export default App;
