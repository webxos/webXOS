import React, { useState, useEffect } from 'react';
import { createSession, connectWallet, signMessage, disconnectWallet } from '../../server/mcp/functions/auth';
import styles from './chatbot.module.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

const ChatbotPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [walletStatus, setWalletStatus] = useState<string>('Disconnected');
  const [userId] = useState<string>('test_user'); // Replace with actual user auth

  useEffect(() => {
    const initializeSession = async () => {
      try {
        await createSession(userId);
        setMessages([{ id: '1', text: 'Welcome to Vial MCP Chatbot! Connect your wallet to begin.', sender: 'bot' }]);
      } catch (err: any) {
        setError(`Failed to initialize session: ${err.message}\n${err.stack}`);
        console.error('Chatbot error:', err);
      }
    };
    initializeSession();
  }, [userId]);

  const handleWalletAction = async (action: 'connect' | 'sign' | 'disconnect') => {
    try {
      if (action === 'connect') {
        const address = await connectWallet(userId);
        setWalletStatus(`Connected: ${address}`);
        setMessages([...messages, { id: Date.now().toString(), text: `Wallet connected: ${address}`, sender: 'bot' }]);
      } else if (action === 'sign') {
        const signature = await signMessage(userId, 'Test message');
        setMessages([...messages, { id: Date.now().toString(), text: `Signed message: ${signature}`, sender: 'bot' }]);
      } else {
        await disconnectWallet(userId);
        setWalletStatus('Disconnected');
        setMessages([...messages, { id: Date.now().toString(), text: 'Wallet disconnected', sender: 'bot' }]);
      }
    } catch (err: any) {
      setError(`Wallet action failed: ${err.message}\n${err.stack}`);
      console.error('Wallet error:', err);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    try {
      const newMessage = { id: Date.now().toString(), text: input, sender: 'user' as 'user' };
      setMessages([...messages, newMessage]);
      setInput('');

      const response = await fetch('/mcp', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'mcp.executeWorkflow',
          params: { user_id: userId, input: input },
          id: 1,
        }),
      });
      const data = await response.json();
      if (data.error) throw new Error(`${data.error.message}\n${data.error.data?.traceback || ''}`);
      setMessages([...messages, newMessage, { id: Date.now().toString(), text: data.result.output, sender: 'bot' }]);
    } catch (err: any) {
      setError(`Failed to process message: ${err.message}\n${err.stack}`);
      console.error('Chatbot error:', err);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Vial MCP Chatbot</h1>
      {error && (
        <div className={styles.error}>
          <pre>{error}</pre>
        </div>
      )}
      <div className={styles.walletStatus}>
        <p>Wallet Status: {walletStatus}</p>
        <div className={styles.buttonGroup}>
          <button onClick={() => handleWalletAction('connect')} className={styles.button}>Connect Wallet</button>
          <button onClick={() => handleWalletAction('sign')} className={styles.button}>Sign Message</button>
          <button onClick={() => handleWalletAction('disconnect')} className={styles.button}>Disconnect Wallet</button>
        </div>
      </div>
      <div className={styles.chatWindow}>
        {messages.map((msg) => (
          <div key={msg.id} className={`${styles.message} ${msg.sender === 'user' ? styles.userMessage : styles.botMessage}`}>
            <p>{msg.text}</p>
          </div>
        ))}
      </div>
      <div className={styles.inputContainer}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          className={styles.input}
          placeholder="Type your message..."
        />
        <button onClick={handleSend} className={styles.button}>Send</button>
      </div>
    </div>
  );
};

export default ChatbotPage;
