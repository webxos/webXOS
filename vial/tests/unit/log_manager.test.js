const { expect } = require('chai');
const { validateLog, compressLog, decompressLog } = require('../src/tools/log_manager');

describe('Log Manager', () => {
  it('should validate valid log data', () => {
    const logData = {
      timestamp: new Date().toISOString(),
      event_type: 'system',
      message: 'Test log',
      metadata: {},
      urgency: 'INFO'
    };
    expect(validateLog(logData)).to.be.true;
  });

  it('should reject invalid log data', () => {
    const logData = { event_type: 'invalid' };
    expect(validateLog(logData)).to.be.false;
  });

  it('should compress and decompress log', () => {
    const logData = {
      timestamp: new Date().toISOString(),
      event_type: 'system',
      message: 'Test log',
      metadata: {},
      urgency: 'INFO'
    };
    const compressed = compressLog(logData);
    const decompressed = decompressLog(compressed);
    expect(decompressed).to.deep.equal(logData);
  });
});

// Rebuild Instructions: Place in /vial/tests/unit/. Install dependencies: `npm install mocha chai --save-dev`. Run `npx mocha tests/unit/log_manager.test.js`. Check /vial/errorlog.md for issues.
