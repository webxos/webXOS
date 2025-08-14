// main/app/chatbot/page.tsx
'use client';

import { useState, useEffect, useCallback } from 'react';
import { trace } from '@opentelemetry/api';
import styles from './chatbot.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';
const PROVIDERS = ['gemini', 'openai', 'claude', 'oci'];

interface Message {
  role: 'user' | 'assistant';
  content: string;
  provider?: string;
}

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([{ role: 'assistant', content: 'Welcome to Vial MCP Chatbot. Select a provider and start chatting.' }]);
  const [input, setInput] = useState('');
  const [provider, setProvider] = useState(PROVIDERS[0]);
  const [isStreaming, setIsStreaming] = useState(false);
  const tracer = trace.getTracer('vial_mcp_chatbot');

  const fetchWithRetry = useCallback(async (url: string, options: RequestInit, retries = 3, baseDelay = 1000): Promise<{ ok: boolean; json: () => Promise<any> }> => {
    const span = tracer.startSpan('fetch_with_retry', { attributes: { url, retries } });
    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, options);
        span.setStatus({ code: 1 });
        span.end();
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        return { ok: response.ok, json: () => response.json() };
      } catch (error: any) {
        if (i === retries - 1) {
          span.recordException(error);
          span.setStatus({ code: 2 });
          span.end();
          setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error.message}`, provider }]);
          return { ok: false, json: () => ({ error: 'Offline mode' }) };
        }
        await new Promise(resolve => setTimeout(resolve, baseDelay * Math.pow(2, i)));
      }
    }
    span.end();
    return { ok: false, json: () => ({ error: 'Fetch failed' }) };
  }, [tracer, provider]);

  const handleQuery = useCallback(async () => {
    if (!input.trim()) return;
    const span = tracer.startSpan('handle_query', { attributes: { provider, query: input } });
    setMessages(prev => [...prev, { role: 'user', content: input }]);
    setInput('');
    setIsStreaming(true);

    try {
      const eventSource = new EventSource(`${API_BASE}/ai/stream?provider=${provider}&query=${encodeURIComponent(input)}`);
      let response = '';

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'content') {
          response += data.content;
          setMessages(prev => [...prev.slice(0, -1), { role: 'assistant', content: response, provider }]);
        } else if (data.type === 'done') {
          setIsStreaming(false);
          eventSource.close();
          span.setStatus({ code: 1 });
          span.end();
        } else if (data.type === 'error') {
          setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${data.error}`, provider }]);
          setIsStreaming(false);
          eventSource.close();
          span.setStatus({ code: 2 });
          span.end();
        }
      };

      eventSource.onerror = () => {
        setMessages(prev => [...prev, { role: 'assistant', content: 'Streaming error occurred', provider }]);
        setIsStreaming(false);
        eventSource.close();
        span.setStatus({ code: 2 });
        span.end();
      };
    } catch (error: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error.message}`, provider }]);
      setIsStreaming(false);
      span.setStatus({ code: 2 });
      span.end();
    }
  }, [input, provider, tracer]);

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Chatbot</h1>
      <select
        value={provider}
        onChange={(e) => setProvider(e.target.value)}
        className={styles.providerSelect}
        disabled={isStreaming}
      >
        {PROVIDERS.map(p => (
          <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
        ))}
      </select>
      <div className={styles.chatWindow}>
        {messages.map((msg, index) => (
          <div key={index} className={`${styles.message} ${msg.role === 'user' ? styles.user : styles.assistant}`}>
            <span className={styles.messageContent}>{msg.content}</span>
            {msg.provider && <span className={styles.providerTag}>{msg.provider}</span>}
          </div>
        ))}
      </div>
      <div className={styles.inputGroup}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your query..."
          className={styles.input}
          disabled={isStreaming}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleQuery();
            }
          }}
        />
        <button onClick={handleQuery} disabled={isStreaming} className={styles.button}>
          {isStreaming ? 'Streaming...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
