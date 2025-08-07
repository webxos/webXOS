/**
 * Minimal JWT decode function for Vial MCP
 * Decodes JWT payload to extract fields like 'exp'
 * No external dependencies, compatible with vial.html
 */
function jwt_decode(token) {
    try {
        // Split JWT into header, payload, and signature
        const parts = token.split('.');
        if (parts.length !== 3) {
            throw new Error('Invalid JWT format');
        }
        // Decode base64url-encoded payload
        const payload = parts[1]
            .replace(/-/g, '+')
            .replace(/_/g, '/');
        // Pad base64 string if needed
        const padded = payload.padEnd(payload.length + (4 - payload.length % 4) % 4, '=');
        // Decode base64 to JSON string and parse
        const decoded = atob(padded);
        return JSON.parse(decoded);
    } catch (err) {
        console.error('JWT Decode Error:', err.message);
        return { exp: 0 }; // Fallback to expired token
    }
}

// Export for global use in vial.html
window.jwt_decode = jwt_decode;
