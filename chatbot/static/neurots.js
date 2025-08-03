const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
if (!ctx) {
    console.error('Failed to get 2D canvas context');
    throw new Error('Canvas context not available');
}

let agents = [
    { name: 'Agent1', color: '#00FF00', x: 0, y: 0, baseX: 0, baseY: 0, active: false },
    { name: 'Agent2', color: '#00CCFF', x: 0, y: 0, baseX: 0, baseY: 0, active: false },
    { name: 'Agent3', color: '#FF00FF', x: 0, y: 0, baseX: 0, baseY: 0, active: false },
    { name: 'Agent4', color: '#FFFF00', x: 0, y: 0, baseX: 0, baseY: 0, active: false }
];
let mouse = { x: canvas.width / 2, y: canvas.height / 2 }; // Initialize to center

function initNeurots() {
    // Set canvas size
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        // Set base positions for agents
        agents.forEach((agent, index) => {
            agent.baseX = canvas.width / 2 + (index - 1.5) * 50; // Spread horizontally
            agent.baseY = canvas.height / 2 + (index - 1.5) * 25; // Spread vertically
            agent.x = agent.baseX;
            agent.y = agent.baseY;
        });
    }
    resizeCanvas();

    // Animation loop
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const time = Date.now() * 0.001;

        agents.forEach((agent, index) => {
            ctx.beginPath();
            ctx.fillStyle = agent.color;
            let radius = 5;

            if (agent.active) {
                // Orbit around mouse
                const orbitRadius = 30;
                const angle = time * 2 + index * Math.PI / 2; // Unique phase per agent
                agent.x = mouse.x + Math.cos(angle) * orbitRadius;
                agent.y = mouse.y + Math.sin(angle) * orbitRadius;
                radius = 7;
            } else {
                // Pulse at base position
                agent.x = agent.baseX;
                agent.y = agent.baseY;
                radius = 5 + Math.sin(time * 2) * 1; // Pulsing effect
            }

            // Draw dot
            ctx.arc(agent.x, agent.y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.closePath();

            // Draw glow
            ctx.beginPath();
            ctx.fillStyle = agent.color + '33'; // 20% opacity
            ctx.arc(agent.x, agent.y, radius + 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.closePath();
        });

        requestAnimationFrame(animate);
    }
    animate();

    // Mouse and touch events
    function updateMousePosition(e) {
        const rect = canvas.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
    }

    function updateTouchPosition(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        mouse.x = touch.clientX - rect.left;
        mouse.y = touch.clientY - rect.top;
    }

    window.addEventListener('mousemove', updateMousePosition);
    window.addEventListener('touchmove', updateTouchPosition, { passive: false });
    window.addEventListener('resize', resizeCanvas);
}

function setAgentsActive(all = false, specificAgent = null) {
    agents.forEach(agent => {
        agent.active = all || (specificAgent && agent.name.toLowerCase() === specificAgent);
    });
}
