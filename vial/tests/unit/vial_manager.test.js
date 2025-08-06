const { expect } = require('chai');
const { validateVial, createVial } = require('../src/tools/vial_manager');

describe('Vial Manager', () => {
  it('should validate valid vial data', () => {
    const vialData = {
      id: 'vial_123456',
      code: { js: 'console.log("test")' },
      training: { model: 'default', epochs: 5 },
      status: 'running',
      latencyHistory: [50.2],
      filePath: '/vial/uploads/vial123456.js',
      createdAt: new Date().toISOString(),
      codeLength: 20
    };
    expect(validateVial(vialData)).to.be.true;
  });

  it('should reject invalid vial data', () => {
    const vialData = { id: 'invalid' };
    expect(validateVial(vialData)).to.be.false;
  });

  it('should create a vial', () => {
    const vial = createVial('vial_123456', 'console.log("test")', { model: 'default', epochs: 5 });
    expect(vial.id).to.equal('vial_123456');
    expect(vial.status).to.equal('running');
  });
});

// Rebuild Instructions: Place in /vial/tests/unit/. Install dependencies: `npm install mocha chai --save-dev`. Run `npx mocha tests/unit/vial_manager.test.js`. Check /vial/errorlog.md for issues.
