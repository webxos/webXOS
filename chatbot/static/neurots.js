const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
let dots = [];
let activeAgent = null;
const DOT_COUNT = 30; // Reduced for performance
const MAX_DISTANCE = 80;
let patternSide = 'left'; // Randomly chosen per agent activation

function setCanvasSize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function getAgentColor(agent) {
    const colors = {
        'agent1': '#00cc00', // Deep neon green
        'agent2': '#00ff99', // Bright cyan-green
        'agent3': '#33ff99', // Light green
        'agent4': '#66ffcc' // Mint green
    };
    return colors[agent] || '#00ff00';
}

function createDot(pattern, index) {
    const base = { radius: Math.random() * 2 + 1, color: activeAgent ? getAgentColor(activeAgent) : '#00ff00' };
    const sideOffset = patternSide === 'left' ? 50 : canvas.width - 150; // Position on left or right
    switch (pattern) {
        case 'spiral': // Agent1: Spiral pattern
            const angle = index * 0.4;
            const radius = 20 + index * 5;
            return {
                ...base,
                x: sideOffset + radius * Math.cos(angle),
                y: canvas.height / 2 + radius * Math.sin(angle),
                vx: Math.cos(angle) * 0.5,
                vy: Math.sin(angle) * 0.5,
                angle: angle,
                radiusSpeed: 0.02
            };
        case 'grid': // Agent2: Grid pattern
            const gridX = (index % 5) * 20;
            const gridY = Math.floor(index / 5) * 20;
            return {
                ...base,
                x: sideOffset + gridX,
                y: canvas.height / 2 - 50 + gridY,
                vx: (Math.random() - 0.5) * 0.2,
                vy: (Math.random() - 0.5) * 0.2
            };
        case 'wave': // Agent3: Wave pattern
            const waveX = index * 10;
            return {
                ...base,
                x: sideOffset + waveX,
                y: canvas.height / 2 + Math.sin(index * 0.5) * 30,
                vx: 0,
                vy: Math.cos(index * 0.5) * 0.5
            };
        case 'cluster': // Agent4: Cluster pattern
            const clusterAngle = Math.random() * Math.PI * 2;
            const clusterRadius = Math.random() * 30;
            return {
                ...base,
                x: sideOffset + clusterRadius * Math.cos(clusterAngle),
                y: canvas.height / 2 + clusterRadius * Math.sin(clusterAngle),
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3
            };
        default: // Default random pattern
            return {
                ...base,
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2
            };
    }
}

function initDots() {
    dots = [];
    const pattern = activeAgent ? {
        'agent1': 'spiral',
        'agent2': 'grid',
        'agent3': 'wave',
        'agent4': 'cluster'
    }[activeAgent] : 'random';
    for (let i = 0; i < DOT_COUNT; i++) {
        dots.push(createDot(pattern, i));
    }
}

function drawDots() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    dots.forEach(dot => {
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
        ctx.fillStyle = dot.color;
        ctx.fill();
    });

    for (let i = 0; i < dots.length; i++) {
        for (let j = i + 1; j < dots.length; j++) {
            const dx = dots[i].x - dots[j].x;
            const dy = dots[i].y - dots[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < MAX_DISTANCE) {
                ctx.beginPath();
                ctx.moveTo(dots[i].x, dots[i].y);
                ctx.lineTo(dots[j].x, dots[j].y);
                ctx.strokeStyle = `rgba(${parseInt(dot.color.slice(1, 3), 16)}, ${parseInt(dot.color.slice(3, 5), 16)}, ${parseInt(dot.color.slice(5, 7), 16)}, ${1 - distance / MAX_DISTANCE})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
}

function updateDots() {
    dots.forEach(dot => {
        if (activeAgent) {
            if (activeAgent === 'agent1') { // Spiral spin
                dot.angle += dot.radiusSpeed;
                const radius = 20 + dot.radius * 5;
                dot.x = (patternSide === 'left' ? 50 : canvas.width - 150) + radius * Math.cos(dot.angle);
                dot.y = canvas.height / 2 + radius * Math.sin(dot.angle);
            } else if (activeAgent === 'agent2') { // Grid oscillation
                dot.x += dot.vx;
                dot.y += dot.vy;
                if (dot.x < (patternSide === 'left' ? 50 : canvas.width - 150) || dot.x > (patternSide === 'left' ? 150 : canvas.width - 50)) dot.vx *= -1;
                if (dot.y < canvas.height / 2 - 50 || dot.y > canvas.height / 2 + 50) dot.vy *= -1;
            } else if (activeAgent === 'agent3') { // Wave motion
                dot.y = canvas.height / 2 + Math.sin((dot.x - (patternSide === 'left' ? 50 : canvas.width - 150)) * 0.05 + Date.now() * 0.001) * 30;
            } else if (activeAgent === 'agent4') { // Cluster pulse
                dot.x += dot.vx;
                dot.y += dot.vy;
                if (Math.abs(dot.x - (patternSide === 'left' ? 50 : canvas.width - 150)) > 50 || Math.abs(dot.y - canvas.height / 2) > 50) {
                    dot.vx *= -1;
                    dot.vy *= -1;
                }
            }
        } else {
            dot.x += dot.vx;
            dot.y += dot.vy;
            if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
            if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
        }
    });
}

function animate() {
    updateDots();
    drawDots();
    requestAnimationFrame(animate);
}

window.setAgentsActive = function(active, agent = null) {
    activeAgent = active ? agent : null;
    patternSide = active ? (Math.random() > 0.5 ? 'left' : 'right') : 'left';
    initDots();
};

window.initNeurots = function() {
    setCanvasSize();
    initDots();
    animate();
    window.addEventListener('resize', setCanvasSize);
};
