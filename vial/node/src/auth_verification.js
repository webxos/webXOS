const jwt = require('jsonwebtoken');
const fs = require('fs');
const path = require('path');

class AuthVerificationAgent {
  constructor(secret = process.env.JWT_SECRET || 'supersecretkey') {
    this.secret = secret;
  }

  verifyToken(token) {
    try {
      const payload = jwt.verify(token, this.secret);
      return { status: 'success', payload };
    } catch (e) {
      console.error(`Token verification error: ${e.message}`);
      fs.appendFileSync(path.resolve(__dirname, '../../vial/errorlog.md'), `- **[${new Date().toISOString()}]** Token verification error: ${e.message}\n`);
      throw new Error(e.message);
    }
  }
}

module.exports = AuthVerificationAgent;
