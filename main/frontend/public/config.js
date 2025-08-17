export const config = {
  API_BASE_URL: window.location.hostname.includes('netlify.app') 
    ? '/.netlify/functions' 
    : window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
      ? 'http://localhost:8888/.netlify/functions' 
      : '/api/mock',
  DEMO_MODE: window.location.hostname === 'localhost' ? false : window.location.hostname.includes('demo') ? true : false,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 2000,
  VIALS: ['vial1', 'vial2', 'vial3', 'vial4'],
  DEFAULT_BALANCE: 38940.0000,
  DEFAULT_REPUTATION: 1200983581,
  DEFAULT_USER_ID: 'a1d57580-d88b-4c90-a0f8-6f2c8511b1e4',
  DEFAULT_WALLET_ADDRESS: 'e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d'
};
