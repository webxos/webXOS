```jsx
import React from 'react';

function VialStats({ vials, wallet }) {
  return (
    <div id="vial-stats">
      <div className="progress-container">
        <span className="progress-label">Wallet</span>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: wallet.balance > 0 ? '100%' : '0%' }}></div>
        </div>
        <span className="status-text">{wallet.balance.toFixed(4)} $WEBXOS | {wallet.hashRate} hashes</span>
      </div>
      {vials.map(vial => (
        <div key={vial.id} className="progress-container">
          <span className="progress-label">{vial.id}</span>
          <div className="progress-bar">
            <div className={`progress-fill ${vial.status === 'stopped' ? 'offline' : ''}`} style={{ width: vial.status === 'running' ? '100%' : '0%' }}></div>
          </div>
          <span className="status-text">{vial.status} | {vial.codeLength} bytes | {vial.tasks.join(', ') || 'none'}</span>
        </div>
      ))}
    </div>
  );
}

export default VialStats;
```
