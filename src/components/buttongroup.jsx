import React from 'react';
import { Button, ButtonGroup as BSButtonGroup } from 'react-bootstrap';
import { LazyLoadComponent } from 'react-lazy-load';

const ButtonGroup = ({ onTroubleshoot, onOAuth, onDashboard, isAuthenticated }) => {
  return (
    <LazyLoadComponent>
      <BSButtonGroup className="my-3 d-flex justify-content-center flex-wrap gap-2">
        <Button variant="success" onClick={onTroubleshoot} aria-label="Run system diagnostics">
          Troubleshoot
        </Button>
        <Button variant="success" onClick={onOAuth} aria-label="Authenticate via OAuth">
          OAuth
        </Button>
        <Button
          variant="success"
          onClick={onDashboard}
          disabled={!isAuthenticated}
          aria-label="Access dashboard"
        >
          Dashboard
        </Button>
      </BSButtonGroup>
    </LazyLoadComponent>
  );
};

export default ButtonGroup;
