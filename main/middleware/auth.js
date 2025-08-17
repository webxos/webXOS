const jwt = require('jsonwebtoken');
const logger = require('../utils/logger');

module.exports.authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) {
        logger.error('No token provided');
        return res.status(401).json({ error: 'No token provided' });
    }
    jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
        if (err) {
            logger.error('Token verification failed', { error: err.message });
            return res.status(403).json({ error: 'Invalid token' });
        }
        req.user = user;
        next();
    });
};
