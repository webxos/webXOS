async function sendMessage() {
    const input = document.getElementById("userInput").value;
    const messages = document.getElementById("messages");
    messages.innerHTML += `<p><b>You:</b> ${input}</p>`;
    const response = await fetch("http://localhost:8000", {  // Update to backend URL
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input })
    });
    const data = await response.json();
    messages.innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
    document.getElementById("userInput").value = "";
    messages.scrollTop = messages.scrollHeight;
}

async function searchQuery() {
    const query = document.getElementById("userInput").value;
    const messages = document.getElementById("messages");
    messages.innerHTML += `<p><b>You (Search):</b> ${query}</p>`;
    try {
        // Search webxos.netlify.app (CORS may require proxy)
        const webResponse = await fetch("https://webxos.netlify.app");
        const webText = await webResponse.text();
        if (webText.toLowerCase().includes(query.toLowerCase())) {
            messages.innerHTML += `<p><b>Search Result:</b> Found '${query}' in WebXOS site: ${webText.slice(0, 100)}...</p>`;
        } else {
            // Search GitHub README
            const githubResponse = await fetch("https://raw.githubusercontent.com/webxos/webxos/main/README.md");
            const githubText = await githubResponse.text();
            if (githubText.toLowerCase().includes(query.toLowerCase())) {
                messages.innerHTML += `<p><b>Search Result:</b> Found '${query}' in WebXOS GitHub README: ${githubText.slice(0, 100)}...</p>`;
            } else {
                messages.innerHTML += `<p><b>Search Result:</b> No results found.</p>`;
            }
        }
    } catch (e) {
        messages.innerHTML += `<p><b>Search Error:</b> ${e.message}</p>`;
    }
    document.getElementById("userInput").value = "";
    messages.scrollTop = messages.scrollHeight;
}