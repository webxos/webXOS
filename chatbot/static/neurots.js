const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
let agents = [
    { name: 'Agent1', color: '#00FF00', x: 0, y: 0, active: false },
    { name: 'Agent2', color: '#00CCFF', x: 0, y: 0, active: false },
    { name: 'Agent3', color: '#FF00FF', x: 0, y: 0, active: false },
    { name: 'Agent4', color: '#FFFF00', x: 0, y: 0, active: false }
];
let mouse = { x: 0, y: 0 };

function initNeurots() {
    // Set canvas size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Initial positions
    agents.forEach((agent, index) => {
        agent.x = canvas.width / 2 + (index * 40 - 60);
        agent.y = canvas.height / 2 + (index * 20 - 30);
    });

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
                const orbitRadius = 20;
                const offsetX = Math.cos(time + index) * orbitRadius;
                const offsetY = Math.sin(time + index) * orbitRadius;
                agent.x = mouse.x + offsetX;
                agent.y = mouse.y + offsetY;
                radius = 7; // Larger when active
            } else {
                // Pulse at fixed position
                agent.x = canvas.width / 2 + (index * 40 - 60);
                agent.y = canvas.height / 2 + (index * 20 - 30);
                radius = 5 + Math.sin(time) * 1; // Pulsing effect
            }

            ctx.arc(agent.x, agent.y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.closePath();

            // Glow effect
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
    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });

    window.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        mouse.x = touch.clientX;
        mouse.y = touch.clientY;
    }, { passive: false });

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        agents.forEach((agent, index) => {
            agent.x = canvas.width / 2 + (index * 40 - 60);
            agent.y = canvas.height / 2 + (index * 20 - 30);
        });
    });
}

function setAgentsActive(all = false, specificAgent = null) {
    agents.forEach(agent => {
        if (all) {
            agent.active = true;
        } else if (specificAgent && agent.name.toLowerCase() === specificAgent) {
            agent.active = true;
        } else {
            agent.active = false;
        }
    });
}
