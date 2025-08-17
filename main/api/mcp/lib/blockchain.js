const crypto = require('crypto');
const logger = require('./logger');

class Blockchain {
  constructor() {
    this.difficulty = 4; // PoW difficulty
  }

  generateHash(data, nonce, previousHash) {
    return crypto.createHash('sha256')
      .update(`${data}${nonce}${previousHash}`)
      .digest('hex');
  }

  mineBlock(walletAddress, vialId, work) {
    try {
      const previousBlock = { hash: '0'.repeat(64) }; // Simplified for demo
      let nonce = 0;
      let hash;
      const data = JSON.stringify({ walletAddress, vialId, work });

      do {
        hash = this.generateHash(data, nonce, previousBlock.hash);
        nonce++;
      } while (!hash.startsWith('0'.repeat(this.difficulty)));

      const block = {
        id: `block_${Date.now()}`,
        previousHash: previousBlock.hash,
        hash,
        data: { walletAddress, vialId, work },
        nonce
      };

      logger.info(`Block mined: ${block.id} for vial ${vialId}`);
      return block;
    } catch (err) {
      logger.error(`Block mining error: ${err.message}`);
      throw err;
    }
  }

  validateBlock(block) {
    try {
      const hash = this.generateHash(
        JSON.stringify(block.data),
        block.nonce,
        block.previousHash
      );
      const isValid = hash === block.hash && hash.startsWith('0'.repeat(this.difficulty));
      logger.info(`Block validation: ${block.id} - ${isValid ? 'Valid' : 'Invalid'}`);
      return isValid;
    } catch (err) {
      logger.error(`Block validation error: ${err.message}`);
      return false;
    }
  }
}

module.exports = { Blockchain };
