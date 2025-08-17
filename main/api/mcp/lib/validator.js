const logger = require('./logger');

function validateUserData(user) {
  try {
    const requiredFields = ['user_id', 'wallet_address', 'balance', 'reputation'];
    const missing = requiredFields.filter(field => !user.hasOwnProperty(field) || user[field] === null);
    if (missing.length) {
      logger.error(`Invalid user data: missing fields - ${missing.join(', ')}`);
      return false;
    }
    if (user.balance < 0 || user.reputation < 0) {
      logger.error(`Invalid user data: negative balance or reputation`);
      return false;
    }
    logger.info(`User data validated: ${user.user_id}`);
    return true;
  } catch (error) {
    logger.error(`User validation error: ${error.message}`);
    return false;
  }
}

function validateVialData(vial) {
  try {
    const requiredFields = ['id', 'user_id', 'status', 'balance'];
    const validStatuses = ['Stopped', 'Training', 'Running'];
    const missing = requiredFields.filter(field => !vial.hasOwnProperty(field) || vial[field] === null);
    if (missing.length) {
      logger.error(`Invalid vial data: missing fields - ${missing.join(', ')}`);
      return false;
    }
    if (!validStatuses.includes(vial.status)) {
      logger.error(`Invalid vial status: ${vial.status}`);
      return false;
    }
    if (vial.balance < 0) {
      logger.error(`Invalid vial data: negative balance`);
      return false;
    }
    logger.info(`Vial data validated: ${vial.id}`);
    return true;
  } catch (error) {
    logger.error(`Vial validation error: ${error.message}`);
    return false;
  }
}

module.exports = { validateUserData, validateVialData };
