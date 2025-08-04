function initNLP() {
    try {
        if (typeof nlp === 'undefined') {
            throw new Error('Compromise not loaded');
        }
        console.log('NLP initialized with Compromise');
        return nlp;
    } catch (error) {
        console.error('NLP initialization failed:', error);
    }
}

export { initNLP };
