// main/app/tools/page.tsx
'use client';

import { useState, useEffect } from 'react';
import styles from './tools.module.css';
import { simulateQuantumCircuit } from '../../server/mcp/functions/quantum';
import { getWalletBalance } from '../../server/mcp/functions/wallet';
import { login } from '../../server/mcp/functions/auth';

export default function Tools() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('apiKey'));
  const [quantumResult, setQuantumResult] = useState(null);
  const [walletBalance, setWalletBalance] = useState(null);

  useEffect(() => {
    if (isAuthenticated) {
      fetchWalletBalance();
    }
  }, [isAuthenticated]);

  const fetchWalletBalance = async () => {
    try {
      const userId = localStorage.getItem('userId');
      const balance = await getWalletBalance(userId);
      setWalletBalance(balance);
    } catch (error) {
      console.error('Failed to fetch wallet balance:', error);
      alert('Failed to load wallet balance.');
    }
  };

  const handleQuantumSimulate = async () => {
    try {
      const circuitData = { num_qubits: 2, gates: ['H', 'CNOT'] };
      const result = await simulateQuantumCircuit('vial123', circuitData);
      setQuantumResult(result);
      alert('Quantum simulation successful!');
    } catch (error) {
      console.error('Quantum simulation failed:', error);
      alert('Quantum simulation failed.');
    }
  };

  const handleAuthenticate = async () => {
    try {
      const username = prompt('Enter username:');
      const password = prompt('Enter password:');
      if (username && password) {
        await login(username, password);
        setIsAuthenticated(true);
        alert('Authentication successful!');
        fetchWalletBalance();
      }
    } catch (error) {
      console.error('Authentication failed:', error);
      alert('Authentication failed.');
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>WebXOS Tools with Vial</h1>
      </header>
      <main className={styles.main}>
        <div className={styles.toolsContainer}>
          {isAuthenticated ? (
            <>
              <h2>Tools</h2>
              <div className={styles.toolSection}>
                <h3>Quantum Simulator</h3>
                <button className={styles.button} onClick={handleQuantumSimulate}>
                  Run Quantum Simulation
                </button>
                {quantumResult && (
                  <div className={styles.result}>
                    <p>Quantum Result: {JSON.stringify(quantumResult)}</p>
                  </div>
                )}
              </div>
              <div className={styles.toolSection}>
                <h3>Wallet</h3>
                <p>Balance: {walletBalance ? `${walletBalance.balance} ETH` : 'Loading...'}</p>
                <button className={styles.button} onClick={fetchWalletBalance}>
                  Refresh Balance
                </button>
              </div>
            </>
          ) : (
            <button className={styles.button} onClick={handleAuthenticate}>
              Authenticate
            </button>
          )}
        </div>
      </main>
      <footer className={styles.footer}>
        <p>Copyright webXOS 2025</p>
      </footer>
    </div>
  );
}
