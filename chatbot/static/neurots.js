const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
let dots = [];
let activeAgent = null;
const DOT_COUNT = 30; // Reduced for performance
const MAX_DISTANCE = 80;
let patternSide = 'left'; // Randomly chosen per agent activation
let animationPhase = 'none'; // 'dissipate', 'reform', or 'none'
let animationProgress = 0; // 0 to 1 for dissipation/reformation

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
    const base = { 
        radius: Math.random() * 2 + 1, 
        color: activeAgent ? getAgentColor(activeAgent) : '#00ff00',
        vx: 0,
        vy: 0,
        angle: 0,
        radiusSpeed: 0,
        opacity: 1,
        targetX: 0,
        targetY: 0
    };
    const position = Math.random();
    const sideOffset = position < 0.33 ? 50 : position < 0.66 ? canvas.width - 150 : canvas.width / 2; // Left, right, or center
    switch (pattern) {
        case 'helix': // Agent1: Double helix protein
            const helixAngle = index * 0.6;
            const helixRadius = 30 + index * 3;
            return {
                ...base,
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                targetX: sideOffset + helixRadius * Math.cos(helixAngle),
                targetY: canvas.height / 2 + helixRadius * Math.sin(helixAngle) + (index % 2 ? 20 : -20),
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                angle: helixAngle,
                radiusSpeed: 0.03
            };
        case 'lattice': // Agent2: Lattice protein
            const latticeX = (index % 5) * 25;
            const latticeY = Math.floor(index / 5) * 25;
            return {
                ...base,
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                targetX: sideOffset + latticeX - 50,
                targetY: canvas.height / 2 - 50 + latticeY,
                vx: (Math.random() - 0.5) * 0.2,
                vy: (Math.random() - 0.5) * 0.2
            };
        case 'ripple': // Agent3: Ripple protein
            const rippleX = index * 12;
            return {
                ...base,
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                targetX: sideOffset + rippleX - 100,
                targetY: canvas.height / 2 + Math.sin(index * 0.4) * 40,
                vx: 0,
                vy: Math.cos(index * 0.4) * 0.5
            };
        case 'starburst': // Agent4: Starburst protein
            const starAngle = Math.random() * Math.PI * 2;
            const starRadius = Math.random() * 40;
            return {
                ...base,
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                targetX: sideOffset + starRadius * Math.cos(starAngle),
                targetY: canvas.height / 2 + starRadius * Math.sin(starAngle),
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3
            };
        default: // Default random pattern
            return {
                ...base,
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                targetX: Math.random() * canvas.width,
                targetY: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2
            };
    }
}

function initDots() {
    dots = [];
    const pattern = activeAgent ? {
        'agent1': 'helix',
        'agent2': 'lattice',
        'agent3': 'ripple',
        'agent4': 'starburst'
    }[activeAgent] : 'random';
    for (let i = 0; i < DOT_COUNT; i++) {
        dots.push(createDot(pattern, i));
    }
    animationPhase = activeAgent ? 'dissipate' : 'none';
    animationProgress = 0;
}

function drawDots() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    dots.forEach(dot => {
        if (!dot) return;
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
        ctx.fillStyle = dot.color.replace(')', `, ${dot.opacity})`).replace('rgb', 'rgba');
        ctx.fill();
    });

    for (let i = 0; i < dots.length; i++) {
        for (let j = i + 1; j < dots.length; j++) {
            if (!dots[i] || !dots[j]) continue;
            const dx = dots[i].x - dots[j].x;
            const dy = dots[i].y - dots[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < MAX_DISTANCE) {
                ctx.beginPath();
                ctx.moveTo(dots[i].x, dots[i].y);
                ctx.lineTo(dots[j].x, dots[j].y);
                ctx.strokeStyle = `rgba(${parseInt(dots[i].color.slice(1, 3), 16)}, ${parseInt(dots[i].color.slice(3, 5), 16)}, ${parseInt(dots[i].color.slice(5, 7), 16)}, ${(1 - distance / MAX_DISTANCE) * dots[i].opacity})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
}

function updateDots() {
    if (animationPhase === 'dissipate') {
        animationProgress += 0.05;
        dots.forEach(dot => {
            if (!dot) return;
            dot.opacity = Math.max(0, 1 - animationProgress);
            dot.x += (Math.random() - 0.5) * 5; // Scatter during dissipation
            dot.y += (Math.random() - 0.5) * 5;
        });
        if (animationProgress >= 1) {
            animationPhase = 'reform';
            animationProgress = 0;
        }
    } else if (animationPhase === 'reform') {
        animationProgress += 0.02; // Slower reformation for effect
        dots.forEach(dot => {
            if (!dot) return;
            dot.opacity = animationProgress;
            dot.x += (dot.targetX - dot.x) * 0.1; // Move toward target
            dot.y += (dot.targetY - dot.y) * 0.1;
            if (activeAgent) {
                if (activeAgent === 'agent1') { // Helix spin
                    dot.angle += dot.radiusSpeed;
                    const radius = 30 + dot.radius * 3;
                    dot.targetX = (patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2) + radius * Math.cos(dot.angle);
                    dot.targetY = canvas.height / 2 + radius * Math.sin(dot.angle) + (dot.angle % 2 ? 20 : -20);
                } else if (activeAgent === 'agent2') { // Lattice oscillation
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    const baseX = patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2;
                    if (dot.x < baseX - 50 || dot.x > baseX + 50) dot.vx *= -1;
                    if (dot.y < canvas.height / 2 - 50 || dot.y > canvas.height / 2 + 50) dot.vy *= -1;
                } else if (activeAgent === 'agent3') { // Ripple motion
                    dot.targetY = canvas.height / 2 + Math.sin((dot.targetX - (patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2)) * 0.05 + Date.now() * 0.001) * 40;
                } else if (activeAgent === 'agent4') { // Starburst pulse
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    const baseX = patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2;
                    if (Math.abs(dot.x - baseX) > 50 || Math.abs(dot.y - canvas.height / 2) > 50) {
                        dot.vx *= -1;
                        dot.vy *= -1;
                    }
                }
            }
        });
        if (animationProgress >= 1) {
            animationPhase = 'none';
        }
    } else {
        dots.forEach(dot => {
            if (!dot) return;
            if (activeAgent) {
                if (activeAgent === 'agent1') { // Helix spin
                    dot.angle += dot.radiusSpeed;
                    const radius = 30 + dot.radius * 3;
                    dot.x = (patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2) + radius * Math.cos(dot.angle);
                    dot.y = canvas.height / 2 + radius * Math.sin(dot.angle) + (dot.angle % 2 ? 20 : -20);
                } else if (activeAgent === 'agent2') { // Lattice oscillation
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    const baseX = patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2;
                    if (dot.x < baseX - 50 || dot.x > baseX + 50) dot.vx *= -1;
                    if (dot.y < canvas.height / 2 - 50 || dot.y > canvas.height / 2 + 50) dot.vy *= -1;
                } else if (activeAgent === 'agent3') { // Ripple motion
                    dot.y = canvas.height / 2 + Math.sin((dot.x - (patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2)) * 0.05 + Date.now() * 0.001) * 40;
                } else if (activeAgent === 'agent4') { // Starburst pulse
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    const baseX = patternSide === 'left' ? 50 : patternSide === 'right' ? canvas.width - 150 : canvas.width / 2;
                    if (Math.abs(dot.x - baseX) > 50 || Math.abs(dot.y - canvas.height / 2) > 50) {
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
}

function animate() {
    updateDots();
    drawDots();
    requestAnimationFrame(animate);
}

window.setAgentsActive = function(active, agent = null) {
    activeAgent = active ? agent : null;
    patternSide = active ? (Math.random() < 0.33 ? 'left' : Math.random() < 0.66 ? 'right' : 'center') : 'left';
    initDots();
};

window.initNeurots = function() {
    setCanvasSize();
    initDots();
    animate();
    window.addEventListener('resize', setCanvasSize);
};
