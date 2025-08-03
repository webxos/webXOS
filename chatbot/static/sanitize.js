function sanitizeInput(input) {
    if (typeof input !== 'string') return input;
    return input
        .replace(/<[^>]*>/g, '')
        .replace(/script|javascript|on\w+/gi, '')
        .replace(/[;`|\\]/g, '')
        .replace(/[<>"'&]/g, c => ({
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;',
            '&': '&amp;'
        })[c])
        .trim();
}
