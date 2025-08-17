const WS_URL = 'wss://webxos.netlify.app/mcp/ws';

class WebSocketHandler {
  private ws: WebSocket | null = null;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectInterval: number = 5000;

  constructor() {
    this.connect();
    document.addEventListener('auth-success', (event: CustomEvent) => {
      this.connect(event.detail.user_id);
    });
  }

  connect(userId?: string) {
    const accessToken = localStorage.getItem('access_token');
    const sessionId = document.cookie.match(/session_id=([^;]+)/)?.[1];
    if (!accessToken || !sessionId || !userId) {
      document.getElementById('websocket-status').innerText = 'Disconnected';
      return;
    }

    this.ws = new WebSocket(`${WS_URL}?token=${accessToken}&session_id=${sessionId}&user_id=${userId}`);
    
    this.ws.onopen = () => {
      document.getElementById('websocket-status').innerText = 'Connected';
      this.reconnectAttempts = 0;
      this.sendMessage({ type: 'subscribe', channel: 'kpi_updates' });
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'session_expired') {
          document.getElementById('output').innerText = 'Session expired. Please re-authenticate.';
          this.ws?.close();
          this.handleSessionExpiration();
        } else if (data.type === 'transaction_update') {
          document.getElementById('transaction-history').innerHTML += `
            <div class="p-2 border-b">
              <p><strong>Transaction ID:</strong> ${data.transaction_id}</p>
              <p><strong>Amount:</strong> ${data.amount} $WEBXOS</p>
              <p><strong>Destination:</strong> ${data.destination_address}</p>
              <p><strong>Timestamp:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
            </div>
          `;
        } else if (data.type === 'kpi_update') {
          document.getElementById('kpi-auth-success').innerText = data.data.auth_success_rate.toFixed(2);
          document.getElementById('kpi-auth-failures').innerText = data.data.auth_failure_count;
          document.getElementById('kpi-sessions').innerText = data.data.active_sessions;
          document.getElementById('kpi-anomalies').innerText = data.data.anomalies_detected;
        }
      } catch (error) {
        document.getElementById('output').innerText = `WebSocket message error: ${error.message}`;
      }
    };

    this.ws.onclose = (event) => {
      document.getElementById('websocket-status').innerText = `Disconnected: ${event.reason || 'Unknown reason'}`;
      this.handleReconnect(userId);
    };

    this.ws.onerror = (error) => {
      document.getElementById('websocket-status').innerText = `WebSocket Error: ${error.type || 'Unknown error'}`;
      document.getElementById('output').innerText = `WebSocket connection failed. Retrying...`;
      this.ws?.close();
    };
  }

  handleSessionExpiration() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    document.cookie = 'session_id=; Max-Age=0; path=/;';
    document.getElementById('user-id').innerText = 'Not logged in';
    document.getElementById('execute-claude-btn').disabled = true;
    document.getElementById('quantum-link-btn').disabled = true;
    document.getElementById('mine-btn').disabled = true;
    document.getElementById('export-btn').disabled = true;
    document.getElementById('api-credentials-btn').disabled = true;
    document.getElementById('git-push-btn').disabled = true;
    document.getElementById('cash-out-btn').disabled = true;
    document.getElementById('logout-btn').disabled = true;
    document.getElementById('data-erasure-btn').disabled = true;
    document.getElementById('auth-btn').click();
  }

  handleReconnect(userId: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        document.getElementById('websocket-status').innerText = `Reconnecting (Attempt ${this.reconnectAttempts})`;
        this.connect(userId);
      }, this.reconnectInterval);
    } else {
      document.getElementById('websocket-status').innerText = 'Disconnected: Max reconnect attempts reached';
      document.getElementById('output').innerText = 'WebSocket connection failed after max retries. Please re-authenticate.';
      this.handleSessionExpiration();
    }
  }

  sendMessage(message: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      document.getElementById('output').innerText = 'Cannot send message: WebSocket is not connected';
    }
  }
}

const wsHandler = new WebSocketHandler();
export { wsHandler };
