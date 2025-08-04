const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
let dots = [];
let activeAgent = null;
let isRandomMode = false; // Track random mode
let growthFactor = 1; // Controls pattern expansion
const DOT_COUNT = 30; // Number of dots
const MAX_DISTANCE = 80; // Max distance for connecting lines
let patternSide = 'center'; // Center for all patterns
let animationPhase = 'dissipate'; // Start with dissipate for initial load
let animationProgress = 0; // 0 to 1 for animation transitions
let stopRequested = false; // Track stop command
let isInitialLoad = true; // Track if this is the initial page load

function setCanvasSize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    console.log('Canvas resized:', canvas.width, canvas.height);
}

function getAgentColor(agent) {
    const colors = {
        'agent1': '#00cc00', // Deep neon green
        'agent2': '#00ff99', // Bright cyan-green
        'agent3': '#ff33cc', // Neon magenta
        'agent4': '#33ccff' // Neon cyan
    };
    return colors[agent] || '#00ff00';
}

function createDot(pattern, index) {
    const base = { 
        radius: Math.random() * 2 + 1, 
        color: activeAgent ? getAgentColor(activeAgent) : '#00ff00',
        vx: (Math.random() - 0.5) * 2, // Default random movement
        vy: (Math.random() - 0.5) * 2,
        angle: 0,
        radiusSpeed: 0,
        opacity: 1,
        targetX: 0,
        targetY: 0,
        index: index // Store index for patterns
    };
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    let dot;
    switch (pattern) {
        case 'helix': // Agent1: Double helix
            const helixAngle = index * 0.6;
            const helixRadius = (10 + index * 1) * growthFactor;
            dot = {
                ...base,
                x: centerX,
                y: centerY,
                targetX: centerX + helixRadius * Math.cos(helixAngle),
                targetY: centerY + helixRadius * Math.sin(helixAngle) + (index % 2 ? 10 : -10),
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                angle: helixAngle,
                radiusSpeed: 0.03
            };
            break;
        case 'cube': // Agent2: Spinning cube (square, asterisk-like)
            const cubeVertices = [
                // Front face (square)
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                // Back face (square)
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                // Connecting edges (approximating cube)
                [0, -1, -1], [0, 1, -1], [-1, 0, -1], [1, 0, -1],
                [0, -1, 1], [0, 1, 1], [-1, 0, 1], [1, 0, 1]
            ];
            const vertex = cubeVertices[index % cubeVertices.length];
            const cubeSize = 15 * growthFactor; // Match helix/torus initial size
            const isoX = (vertex[0] - vertex[1]) * cubeSize * 0.707; // Isometric projection
            const isoY = (vertex[0] + vertex[1] - 2 * vertex[2]) * cubeSize * 0.5;
            dot = {
                ...base,
                x: centerX,
                y: centerY,
                targetX: centerX + isoX,
                targetY: centerY + isoY,
                vx: (Math.random() - 0.5) * 0.2,
                vy: (Math.random() - 0.5) * 0.2,
                angle: index * 0.4,
                radiusSpeed: 0.02 // Spin speed
            };
            break;
        case 'torus': // Agent3: Torus (ring)
            const torusAngle = index * 0.4;
            const torusRadius = 15 * growthFactor;
            dot = {
                ...base,
                x: centerX,
                y: centerY,
                targetX: centerX + torusRadius * Math.cos(torusAngle),
                targetY: centerY + torusRadius * Math.sin(torusAngle),
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                angle: torusAngle,
                radiusSpeed: 0.02
            };
            break;
        case 'star': // Agent4: Triangular star quantum pattern
            const starAngle = (index % 5) * (2 * Math.PI / 5); // 5-point star
            const starRadius = (index < 15 ? 15 : 10) * growthFactor; // Inner/outer radius
            const offset = index < 15 ? 0 : Math.PI / 5; // Offset for star points
            dot = {
                ...base,
                x: centerX,
                y: centerY,
                targetX: centerX + starRadius * Math.cos(starAngle + offset),
                targetY: centerY + starRadius * Math.sin(starAngle + offset),
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                angle: starAngle,
                radiusSpeed: 0.015 // Slower pulse
            };
            break;
        default: // Default random pattern
            dot = {
                ...base,
                x: centerX,
                y: centerY,
                targetX: centerX + (Math.random() - 0.5) * canvas.width,
                targetY: centerY + (Math.random() - 0.5) * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2
            };
    }
    console.log(`Created dot for ${pattern}:`, dot);
    return dot;
}

function initDots() {
    dots = [];
    const pattern = activeAgent ? {
        'agent1': 'helix',
        'agent2': 'cube',
        'agent3': 'torus',
        'agent4': 'star'
    }[activeAgent] : 'random';
    for (let i = 0; i < DOT_COUNT; i++) {
        const dot = createDot(pattern, i);
        if (dot) dots.push(dot);
    }
    console.log(`Initialized ${dots.length} dots for pattern: ${pattern}, randomMode: ${isRandomMode}, growthFactor: ${growthFactor}`);
    if (isInitialLoad || !activeAgent) {
        animationPhase = 'dissipate'; // Only dissipate on initial load or clear
    } else {
        animationPhase = 'reform'; // Start with reform for agent activation
    }
    animationProgress = 0;
    if (isRandomMode) {
        growthFactor = 1; // Reset growth factor
        stopRequested = false;
    }
}

function checkBounds() {
    // Looser bounds to prevent early stopping
    return dots.some(dot => 
        dot.x < -50 || dot.x > canvas.width + 50 || 
        dot.y < -50 || dot.y > canvas.height + 50
    );
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
    console.log('Drawn dots:', dots.length);
}

function updateDots() {
    if (stopRequested || (isRandomMode && checkBounds())) {
        console.log('Stopping random mode: ', { stopRequested, boundsHit: checkBounds() });
        isRandomMode = false;
        growthFactor = 1;
        animationPhase = 'none';
        initDots();
        return;
    }

    if (animationPhase === 'dissipate') {
        animationProgress += 0.05;
        dots.forEach(dot => {
            if (!dot) return;
            dot.opacity = Math.max(0, 1 - animationProgress);
            dot.x += (Math.random() - 0.5) * 3;
            dot.y += (Math.random() - 0.5) * 3;
        });
        if (animationProgress >= 1) {
            animationPhase = 'reform';
            animationProgress = 0;
            isInitialLoad = false; // End initial load phase
        }
    } else if (animationPhase === 'reform') {
        animationProgress += 0.03;
        dots.forEach(dot => {
            if (!dot) return;
            dot.opacity = animationProgress;
            dot.x += (dot.targetX - dot.x) * 0.1;
            dot.y += (dot.targetY - dot.y) * 0.1;
        });
        if (animationProgress >= 1) {
            animationPhase = 'none';
        }
    } else {
        dots.forEach(dot => {
            if (!dot) return;
            if (activeAgent) {
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                if (activeAgent === 'agent1') { // Helix spin
                    dot.angle += dot.radiusSpeed;
                    const radius = (10 + dot.radius * 1) * growthFactor;
                    dot.targetX = centerX + radius * Math.cos(dot.angle);
                    dot.targetY = centerY + radius * Math.sin(dot.angle) + (dot.angle % 2 ? 10 : -10);
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                } else if (activeAgent === 'agent2') { // Cube spin
                    dot.angle += dot.radiusSpeed;
                    const cubeVertices = [
                        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                        [0, -1, -1], [0, 1, -1], [-1, 0, -1], [1, 0, -1],
                        [0, -1, 1], [0, 1, 1], [-1, 0, 1], [1, 0, 1]
                    ];
                    const vertex = cubeVertices[dot.index % cubeVertices.length];
                    const cubeSize = 15 * growthFactor;
                    const cosA = Math.cos(dot.angle);
                    const sinA = Math.sin(dot.angle);
                    // Rotate around Z-axis for spinning effect
                    const rotX = vertex[0] * cosA - vertex[1] * sinA;
                    const rotY = vertex[0] * sinA + vertex[1] * cosA;
                    const isoX = (rotX - vertex[2]) * cubeSize * 0.707;
                    const isoY = (rotX + vertex[2]) * cubeSize * 0.5;
                    dot.targetX = centerX + isoX;
                    dot.targetY = centerY + isoY;
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                    dot.vx = dot.vx || (Math.random() - 0.5) * 0.2;
                    dot.vy = dot.vy || (Math.random() - 0.5) * 0.2;
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    const bound = 15 * growthFactor;
                    if (Math.abs(dot.x - dot.targetX) > bound) dot.vx *= -0.9;
                    if (Math.abs(dot.y - dot.targetY) > bound) dot.vy *= -0.9;
                } else if (activeAgent === 'agent3') { // Torus rotation
                    dot.angle += dot.radiusSpeed;
                    const radius = 15 * growthFactor;
                    dot.targetX = centerX + radius * Math.cos(dot.angle);
                    dot.targetY = centerY + radius * Math.sin(dot.angle);
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                } else if (activeAgent === 'agent4') { // Star pulse
                    dot.angle += dot.radiusSpeed;
                    const starAngle = (dot.index % 5) * (2 * Math.PI / 5);
                    const offset = dot.index < 15 ? 0 : Math.PI / 5;
                    const starRadius = (dot.index < 15 ? 15 : 10) * growthFactor;
                    dot.targetX = centerX + starRadius * Math.cos(starAngle + offset + dot.angle);
                    dot.targetY = centerY + starRadius * Math.sin(starAngle + offset + dot.angle);
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                    dot.vx = dot.vx || (Math.random() - 0.5) * 0.3;
                    dot.vy = dot.vy || (Math.random() - 0.5) * 0.3;
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    const bound = 15 * growthFactor;
                    if (Math.abs(dot.x - centerX) > bound) dot.vx *= -0.9;
                    if (Math.abs(dot.y - centerY) > bound) dot.vy *= -0.9;
                }
            } else {
                dot.x += dot.vx;
                dot.y += dot.vy;
                if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
            }
        });
        if (isRandomMode) {
            growthFactor += 0.005; // Slow growth for organic effect
            console.log('Random mode: growthFactor =', growthFactor);
        }
    }
}

function animate() {
    updateDots();
    drawDots();
    requestAnimationFrame(animate);
}

window.setAgentsActive = function(active, agent = null, randomMode = false) {
    if (active && activeAgent !== agent) {
        // Only reinitialize if switching agents
        activeAgent = agent;
        isRandomMode = randomMode;
        stopRequested = false;
        patternSide = 'center';
        console.log(`setAgentsActive: agent=${agent}, randomMode=${randomMode}, patternSide=${patternSide}`);
        initDots();
    } else if (!active && activeAgent !== null) {
        // Reset to default random pattern on clear
        activeAgent = null;
        isRandomMode = false;
        stopRequested = false;
        patternSide = 'center';
        console.log('Resetting to default pattern');
        initDots();
    }
};

window.stopRandomMode = function() {
    stopRequested = true;
    isRandomMode = false;
    growthFactor = 1;
    console.log('Random mode stopped');
    initDots();
};

window.initNeurots = function() {
    if (!canvas || !ctx) {
        console.error('Canvas or context not found');
        return;
    }
    setCanvasSize();
    initDots();
    animate();
    window.addEventListener('resize', setCanvasSize);
    console.log('Neurots initialized');
};
