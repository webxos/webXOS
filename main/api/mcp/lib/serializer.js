const logger = require('./logger');

function serializeData(data) {
  try {
    const serialized = JSON.stringify(data, (key, value) => {
      if (typeof value === 'bigint') {
        return value.toString();
      }
      return value;
    });
    logger.info(`Data serialized successfully`);
    return serialized;
  } catch (error) {
    logger.error(`Serialization error: ${error.message}`);
    throw new Error(`Failed to serialize data: ${error.message}`);
  }
}

function deserializeData(data) {
  try {
    const deserialized = JSON.parse(data);
    logger.info(`Data deserialized successfully`);
    return deserialized;
  } catch (error) {
    logger.error(`Deserialization error: ${error.message}`);
    throw new Error(`Failed to deserialize data: ${error.message}`);
  }
}

module.exports = { serializeData, deserializeData };
