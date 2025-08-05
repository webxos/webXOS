// src/pipelines/tokenization.js
exports.tokenize = (text) => {
    try {
        return text.split(' ').map(word => word.toLowerCase());
    } catch (err) {
        console.error(`[TOKENIZATION] Error: ${err.message}`);
        throw err;
    }
};

// Instructions:
// - Tokenizes ML inputs
// - Extend with libraries like sentencepiece
