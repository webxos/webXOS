import searchData from './searchData.json';

// Base URL for links
const BASE_URL = 'https://webxos.netlify.app';

// DOM elements
const messages = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const status = document.getElementById('status');

// Add Enter key support
userInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSearch();
    }
});

// Search function
function handleSearch() {
    const query = userInput.value.trim().toLowerCase();
    if (!query) {
        displayStatus('Please enter a search query.', 'error');
        return;
    }

    displayStatus('Searching...', 'success');
    messages.innerHTML += `<p><strong>You:</strong> ${escapeHTML(query)}</p>`;

    // Filter search data
    const results = searchData.filter(item =>
        item.text.toLowerCase().includes(query) ||
        item.path.toLowerCase().includes(query) ||
        item.source.toLowerCase().includes(query)
    );

    setTimeout(() => {
        if (results.length === 0) {
            messages.innerHTML += `<p><strong>Bot:</strong> No results found for "${escapeHTML(query)}".</p>`;
            displayStatus('No results found.', 'error');
        } else {
            // Display detailed results
            messages.innerHTML += `<p><strong>Bot:</strong> Found ${results.length} result(s) for "${escapeHTML(query)}":</p>`;
            results.forEach(result => {
                const title = extractTitle(result.text) || result.path;
                const snippet = extractSnippet(result.text, query);
                const url = `${BASE_URL}${result.path}`;
                messages.innerHTML += `
                    <p>
                        <strong><a href="${url}" target="_blank">${escapeHTML(title)}</a></strong><br>
                        ${escapeHTML(snippet)}...<br>
                        <a href="${url}" target="_blank">${url}</a>
                    </p>`;
            });

            // Add minimized results list
            messages.innerHTML += `<p><strong>Results Summary:</strong></p><ul>`;
            results.forEach(result => {
                const title = extractTitle(result.text) || result.path;
                const url = `${BASE_URL}${result.path}`;
                messages.innerHTML += `<li><a href="${url}" target="_blank">${escapeHTML(title)}</a></li>`;
            });
            messages.innerHTML += `</ul>`;

            displayStatus(`Found ${results.length} result(s).`, 'success');
        }

        // Scroll to bottom
        messages.scrollTop = messages.scrollHeight;
        userInput.value = '';
    }, 500); // Simulate network delay
}

// Extract title from HTML or Markdown
function extractTitle(text) {
    const match = text.match(/<title>(.*?)<\/title>/i) || text.match(/# (.*?)\n/);
    return match ? match[1] : null;
}

// Extract snippet around query
function extractSnippet(text, query) {
    const index = text.toLowerCase().indexOf(query.toLowerCase());
    if (index === -1) return text.slice(0, 100);
    const start = Math.max(0, index - 50);
    const end = Math.min(text.length, index + query.length + 50);
    return text.slice(start, end);
}

// Escape HTML to prevent XSS
function escapeHTML(str) {
    return str.replace(/[&<>"']/g, match => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[match]));
}

// Display status message
function displayStatus(message, type) {
    status.textContent = message;
    status.className = type;
}

// Neural network animation
function initNeuralNetwork() {
    const canvas = document.getElementById('neuralNetwork');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const dots = [];
    for (let i = 0; i < 10; i++) {
        dots.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2
        });
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#00FF00';
        dots.forEach(dot => {
            dot.x += dot.vx;
            dot.y += dot.vy;
            if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
            if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
            ctx.beginPath();
            ctx.arc(dot.x, dot.y, 5, 0, Math.PI * 2);
            ctx.fill();
        });
        requestAnimationFrame(animate);
    }
    animate();
}

// Register service worker for PWA
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js');
}

initNeuralNetwork();
