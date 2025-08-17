const crypto = require('crypto');
const logger = require('./logger');

const algorithm = 'aes-256-cbc';
const key = Buffer.from(process.env.ENCRYPTION_KEY || crypto.randomBytes(32).toString('hex'), 'hex');
const iv = crypto.randomBytes(16);

function encryptData(data) {
  try {
    const cipher = crypto.createCipheriv(algorithm, key, iv);
    let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
    encrypted += cipher.final('hex');
    logger.info('Data encrypted successfully');
    return { encrypted, iv: iv.toString('hex') };
  } catch (error) {
    logger.error(`Encryption error: ${error.message}`);
    throw new Error(`Failed to encrypt data: ${error.message}`);
  }
}

function decryptData(encryptedData, ivHex) {
  try {
    const iv = Buffer.from(ivHex, 'hex');
    const decipher = crypto.createDecipheriv(algorithm, key, iv);
    let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    logger.info('Data decrypted successfully');
    return JSON.parse(decrypted);
  } catch (error) {
    logger.error(`Decryption error: ${error.message}`);
    throw new Error(`Failed to decrypt data: ${error.message}`);
  }
}

module.exports = { encryptData, decryptData };
