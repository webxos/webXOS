import GUN from 'https://cdn.jsdelivr.net/npm/gun@0.2020/gun.min.js';

const gun = GUN();
function initGun() {
  gun.get('webxos-results').on(data => {
    const messages = document.getElementById('messages');
    messages.innerHTML += `<p>Real-time sync: ${JSON.stringify(data)}</p>`;
  });
}

async function syncResults(agent, query, results) {
  gun.get('webxos-results').put({ [query]: { agent, results, timestamp: Date.now() } });
}

export { initGun, syncResults };
