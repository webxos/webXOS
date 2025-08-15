import React from 'react';
import { LazyLoadComponent } from 'react-lazy-load';
import './Console.css';

const Console = ({ logs }) => {
  return (
    <LazyLoadComponent>
      <div className="console" role="log" aria-live="polite">
        {logs.map((log, index) => (
          <p key={index} dangerouslySetInnerHTML={{ __html: log }} />
        ))}
      </div>
    </LazyLoadComponent>
  );
};

export default Console;
