try {
    console.log('Loading neurots.js');
    const canvas = document.getElementById('neuralCanvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Failed to get 2D canvas context');
        throw new Error('Canvas context not available');
    }

    let agents = [
        { name: 'Agent1', color: '#00FF00', x: 0, y: 0, baseX: 0, baseY: 0, active: false, velocityX: 0, velocityY: 0 },
        { name: 'Agent2', color: '#00CCFF', x: 0, y: 0, baseX: 0, baseY: 0, active: false, velocityX: 0, velocityY: 0 },
        { name: 'Agent3', color: '#FF00FF', x: 0, y: 0, baseX: 0, baseY: 0, active: false, velocityX: 0, velocityY: 0 },
        { name: 'Agent4', color: '#FFFF00', x: 0, y: 0, baseX: 0, baseY: 0, active: false, velocityX: 0, velocityY: 0 }
    ];
    let mouse = { x: canvas.width / 2, y: canvas.height / 2 }; // Initialize to center

    // Simple noise function for organic floating
    function simpleNoise(t, seed) {
        const x = Math.sin(t + seed) * 43758.5453;
        return (x - Math.floor(x)) * 2 - 1; // Range: -1 to 1
    }

    function initNeurots() {
        console.log('Initializing neurots');
        // Set canvas size
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            // Set base positions for agents
            agents.forEach((agent, index) => {
                agent.baseX = canvas.width / 2 + (index - 1.5) * 50;
                agent.baseY = canvas.height / 2 + (index - 1.5) * 25;
                agent.x = agent.baseX;
                agent.y = agent.baseY;
                agent.velocityX = (Math.random() - 0.5) * 2;
                agent.velocityY = (Math.random() - 0.5) * 2;
            });
            console.log('Canvas resized, agents initialized:', agents);
        }
        resizeCanvas();

        // Animation loop
        function animate() {
            try {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const time = Date.now() * 0.001;

                agents.forEach((agent, index) => {
                    ctx.beginPath();
                    ctx.fillStyle = agent.active ? agent.color : '#00FF00';
                    let radius = 5;

                    if (agent.active) {
                        // Orbit around mouse
                        const orbitRadius = 30;
                        const angle = time * 2 + index * Math.PI / 2;
                        agent.x = mouse.x + Math.cos(angle) * orbitRadius;
                        agent.y = mouse.y + Math.sin(angle) * orbitRadius;
                        radius = 7;
                    } else {
                        // Float randomly around base position
                        const noiseX = simpleNoise(time * 0.5, index);
                        const noiseY = simpleNoise(time * 0.5 + 100, index);
                        agent.velocityX += noiseX * 0.1;
                        agent.velocityY += noiseY * 0.1;
                        agent.velocityX = Math.max(-2, Math.min(2, agent.velocityX));
                        agent.velocityY = Math.max(-2, Math.min(2, agent.velocityY));
                        agent.x += agent.velocityX;
                        agent.y += agent.velocityY;
                        agent.x = Math.max(50, Math.min(canvas.width - 50, agent.x));
                        agent.y = Math.max(50, Math.min(canvas.height - 50, agent.y));
                        radius = 5;
                    }

                    // Draw dot
                    ctx.arc(agent.x, agent.y, radius, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.closePath();

                    // Draw glow
                    ctx.beginPath();
                    ctx.fillStyle = (agent.active ? agent.color : '#00FF00') + '33';
                    ctx.arc(agent.x, agent.y, radius + 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.closePath();
                });

                requestAnimationFrame(animate);
            } catch (error) {
                console.error('Animation loop error:', error);
            }
        }
        animate();

        // Mouse and touch events
        function updateMousePosition(e) {
            const rect = canvas.getBoundingClientRect();
            mouse.x = e.clientX - rect.left;
            mouse.y = e.clientY - rect.top;
            console.log('Mouse moved:', mouse);
        }

        function updateTouchPosition(e) {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            mouse.x = touch.clientX - rect.left;
            mouse.y = touch.clientY - rect.top;
            console.log('Touch moved:', mouse);
        }

        window.addEventListener('mousemove', updateMousePosition);
        window.addEventListener('touchmove', updateTouchPosition, { passive: false });
        window.addEventListener('resize', resizeCanvas);
    }

    // Expose initNeurots globally
    window.initNeurots = initNeurots;
    console.log('neurots.js loaded, initNeurots defined');

    function setAgentsActive(all = false, specificAgent = null) {
        agents.forEach(agent => {
            agent.active = all || (specificAgent && agent.name.toLowerCase() === specificAgent);
        });
        console.log('Agents active state:', agents.map(a => `${a.name}: ${a.active}`));
    }
} catch (error) {
    console.error('neurots.js runtime error:', error);
}
