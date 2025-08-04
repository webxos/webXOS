(function () {
    const canvas = document.getElementById('neuralCanvas');
    if (!canvas) {
        console.error('Canvas element with ID "neuralCanvas" not found');
        return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Failed to get 2D context for canvas');
        return;
    }

    let dots = [];
    let activeAgents = [];
    let isDnaMode = false;
    let isGalaxyMode = false;
    let growthFactor = 0.5;
    let dotCount = 30;
    const MAX_DISTANCE = 80;
    const MAX_COLLAB_DISTANCE = 120;
    let animationPhase = 'explode';
    let animationProgress = 0;
    let isInitialLoad = true;

    // Set canvas size
    function setCanvasSize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    // Get agent color
    function getAgentColor(agent) {
        const colors = {
            'agent1': '#00cc00', // Green
            'agent2': '#00ff99', // Cyan
            'agent3': '#ff33cc', // Pink
            'agent4': '#33ccff'  // Blue
        };
        return colors[agent] || '#00ff00';
    }

    // Get random position avoiding chatbox
    function getRandomPosition() {
        const isMobile = window.innerWidth <= 768;
        const textContainer = {
            x: canvas.width * 0.25,
            y: isMobile ? canvas.height * 0.3 : canvas.height * 0.7,
            width: canvas.width * 0.5,
            height: canvas.height * (isMobile ? 0.7 : 0.3)
        };
        let x, y;
        do {
            x = Math.random() * canvas.width;
            y = isMobile ? Math.random() * 100 : Math.random() * canvas.height;
        } while (
            x > textContainer.x && x < textContainer.x + textContainer.width &&
            y > textContainer.y && y < textContainer.y + textContainer.height
        );
        return { x, y };
    }

    // Create dot for pattern
    function createDot(pattern, index, agent) {
        const base = {
            radius: 2,
            color: getAgentColor(agent),
            angle: Math.random() * Math.PI * 2,
            radiusSpeed: 0,
            opacity: 1,
            targetX: 0,
            targetY: 0,
            index: index,
            agent: agent,
            x: canvas.width / 2,
            y: canvas.height / 2,
            vx: (Math.random() - 0.5) * 10, // Initial velocity for explosion
            vy: (Math.random() - 0.5) * 10
        };
        let center = isDnaMode || isGalaxyMode ? getRandomPosition() : { x: canvas.width / 2, y: canvas.height / 2 };
        if (isGalaxyMode) {
            base.color = getAgentColor(index < 4 ? `agent${index + 1}` : `agent${Math.floor(Math.random() * 4) + 1}`);
            base.type = index < 4 ? 'star' : Math.random() < 0.3 ? 'star' : Math.random() < 0.6 ? 'planet' : Math.random() < 0.9 ? 'comet' : 'smallPlanet';
            base.opacity = base.type === 'star' ? Math.random() * 0.5 + 0.5 : 1;
            base.radius = base.type === 'planet' ? 3 + Math.random() * 2 : base.type === 'comet' ? 2 : base.type === 'smallPlanet' ? 1 + Math.random() * 1 : 0.5 + Math.random() * 0.5;
            base.vx = base.type === 'comet' ? (Math.random() - 0.5) * 6 : (Math.random() - 0.5) * 2;
            base.vy = base.type === 'comet' ? (Math.random() - 0.5) * 6 : (Math.random() - 0.5) * 2;
            base.trail = base.type === 'comet' ? [] : null;
            base.orbitCenter = base.type === 'smallPlanet' ? { x: center.x, y: center.y, radius: 10 + Math.random() * 10 } : null;
            base.glow = base.type === 'star' ? Math.random() * 2 + 1 : 0;
        }
        let dot;
        switch (pattern) {
            case 'helix':
                const helixAngle = index * 0.6;
                const helixRadius = 10 * growthFactor;
                dot = {
                    ...base,
                    targetX: center.x + helixRadius * Math.cos(helixAngle),
                    targetY: center.y + helixRadius * Math.sin(helixAngle) + (index % 2 ? 10 : -10),
                    angle: helixAngle,
                    radiusSpeed: 0.03
                };
                break;
            case 'cube':
                const cubeVertices = [
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                ];
                const vertex = cubeVertices[index % cubeVertices.length];
                const cubeSize = 15 * growthFactor;
                const isoX = (vertex[0] - vertex[1]) * cubeSize * 0.707;
                const isoY = (vertex[0] + vertex[1] - 2 * vertex[2]) * cubeSize * 0.5;
                dot = {
                    ...base,
                    targetX: center.x + isoX,
                    targetY: center.y + isoY,
                    angle: index * 0.4,
                    radiusSpeed: 0.02
                };
                break;
            case 'torus':
                const torusAngle = index * 0.4;
                const torusRadius = 15 * growthFactor;
                dot = {
                    ...base,
                    targetX: center.x + torusRadius * Math.cos(torusAngle),
                    targetY: center.y + torusRadius * Math.sin(torusAngle),
                    angle: torusAngle,
                    radiusSpeed: 0.02
                };
                break;
            case 'star':
                const starAngle = (index % 5) * (2 * Math.PI / 5);
                const starRadius = (index < 15 ? 15 : 10) * growthFactor;
                const offset = index < 15 ? 0 : Math.PI / 5;
                dot = {
                    ...base,
                    targetX: center.x + starRadius * Math.cos(starAngle + offset),
                    targetY: center.y + starRadius * Math.sin(starAngle + offset),
                    angle: starAngle,
                    radiusSpeed: 0.015
                };
                break;
            case 'dna':
                const dnaAngle = index * 0.2; // Slower rotation
                const dnaRadius = 20 * growthFactor;
                const strandOffset = (index % 4) * 5; // Four interwoven strands
                const phaseShift = (agent === 'agent1' ? 0 : agent === 'agent2' ? Math.PI / 2 : agent === 'agent3' ? Math.PI : 3 * Math.PI / 2);
                dot = {
                    ...base,
                    targetX: center.x + dnaRadius * Math.cos(dnaAngle + phaseShift) + strandOffset,
                    targetY: center.y + dnaRadius * Math.sin(dnaAngle + phaseShift),
                    angle: dnaAngle,
                    radiusSpeed: 0.005 // Much slower for smooth DNA spirals
                };
                break;
            default:
                dot = {
                    ...base,
                    targetX: center.x + (Math.random() - 0.5) * canvas.width,
                    targetY: center.y + (Math.random() - 0.5) * canvas.height,
                    vx: (Math.random() - 0.5) * 8,
                    vy: (Math.random() - 0.5) * 8
                };
        }
        return dot;
    }

    // Initialize dots
    function initDots() {
        dots = [];
        if (isGalaxyMode) {
            for (let i = 0; i < 4; i++) {
                const dot = createDot('galaxy', i, `agent${i + 1}`);
                if (dot) dots.push(dot);
            }
            dotCount = 4;
        } else if (isDnaMode) {
            activeAgents = ['agent1', 'agent2', 'agent3', 'agent4'];
            activeAgents.forEach(agent => {
                const pattern = 'dna';
                for (let i = 0; i < 30; i++) {
                    const dot = createDot(pattern, i, agent);
                    if (dot) dots.push(dot);
                }
            });
            dotCount = 30 * 4;
        } else {
            const pattern = activeAgents.length > 0 ? {
                'agent1': 'helix',
                'agent2': 'cube',
                'agent3': 'torus',
                'agent4': 'star'
            }[activeAgents[0]] : 'random';
            for (let i = 0; i < 30; i++) {
                const dot = createDot(pattern, i, activeAgents[0] || 'default');
                if (dot) dots.push(dot);
            }
            dotCount = 30;
        }
        animationPhase = isInitialLoad ? 'explode' : 'reform';
        animationProgress = 0;
        growthFactor = 0.5;
        isInitialLoad = false;
    }

    // Draw dots and connections
    function drawDots() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        dots.forEach(dot => {
            if (!dot) return;
            if (dot.glow) {
                ctx.beginPath();
                ctx.arc(dot.x, dot.y, dot.radius + dot.glow, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${parseInt(dot.color.slice(1, 3), 16)}, ${parseInt(dot.color.slice(3, 5), 16)}, ${parseInt(dot.color.slice(5, 7), 16)}, ${dot.opacity * 0.2})`;
                ctx.fill();
            }
            ctx.beginPath();
            ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
            ctx.fillStyle = dot.color.replace(')', `, ${dot.opacity})`).replace('rgb', 'rgba');
            ctx.fill();
            if (dot.type === 'comet' && dot.trail) {
                ctx.beginPath();
                ctx.moveTo(dot.x, dot.y);
                dot.trail.forEach(point => ctx.lineTo(point.x, point.y));
                ctx.strokeStyle = `rgba(${parseInt(dot.color.slice(1, 3), 16)}, ${parseInt(dot.color.slice(3, 5), 16)}, ${parseInt(dot.color.slice(5, 7), 16)}, 0.5)`;
                ctx.stroke();
            }
        });

        const maxDistance = isDnaMode ? MAX_COLLAB_DISTANCE : MAX_DISTANCE;
        for (let i = 0; i < dots.length; i++) {
            for (let j = i + 1; j < dots.length; j++) {
                if (!dots[i] || !dots[j]) continue;
                if (isDnaMode && dots[i].agent !== dots[j].agent) {
                    const dx = dots[i].x - dots[j].x;
                    const dy = dots[i].y - dots[j].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < maxDistance) {
                        ctx.beginPath();
                        ctx.moveTo(dots[i].x, dots[i].y);
                        ctx.lineTo(dots[j].x, dots[j].y);
                        ctx.strokeStyle = `rgba(255, 255, 255, ${(1 - distance / maxDistance) * dots[i].opacity})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                } else if (!isDnaMode && !isGalaxyMode) {
                    const dx = dots[i].x - dots[j].x;
                    const dy = dots[i].y - dots[j].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < maxDistance) {
                        ctx.beginPath();
                        ctx.moveTo(dots[i].x, dots[i].y);
                        ctx.lineTo(dots[j].x, dots[j].y);
                        ctx.strokeStyle = `rgba(${parseInt(dots[i].color.slice(1, 3), 16)}, ${parseInt(dots[i].color.slice(3, 5), 16)}, ${parseInt(dots[i].color.slice(5, 7), 16)}, ${(1 - distance / maxDistance) * dots[i].opacity})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
        }
    }

    // Update dot positions
    function updateDots() {
        dots.forEach(dot => {
            if (!dot) return;
            if (animationPhase === 'explode') {
                dot.x += dot.vx;
                dot.y += dot.vy;
                // Bounce off walls
                if (dot.x < 0 || dot.x > canvas.width) {
                    dot.vx *= -0.8; // Slight energy loss on bounce
                    dot.x = Math.max(0, Math.min(canvas.width, dot.x));
                }
                if (dot.y < 0 || dot.y > canvas.height) {
                    dot.vy *= -0.8;
                    dot.y = Math.max(0, Math.min(canvas.height, dot.y));
                }
                dot.opacity = Math.max(0, dot.opacity - 0.01);
            } else if (isGalaxyMode) {
                if (dot.type === 'star') {
                    dot.opacity = 0.5 + Math.sin(Date.now() * 0.005 + dot.index) * 0.5;
                } else if (dot.type === 'planet') {
                    dot.angle += 0.01;
                    dot.targetX = dot.x + Math.cos(dot.angle) * 10 * growthFactor;
                    dot.targetY = dot.y + Math.sin(dot.angle) * 10 * growthFactor;
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                } else if (dot.type === 'comet') {
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                    if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                    dot.trail.push({ x: dot.x, y: dot.y });
                    if (dot.trail.length > 8) dot.trail.shift();
                } else if (dot.type === 'smallPlanet') {
                    dot.angle += 0.02;
                    dot.targetX = dot.orbitCenter.x + Math.cos(dot.angle) * dot.orbitCenter.radius * growthFactor;
                    dot.targetY = dot.orbitCenter.y + Math.sin(dot.angle) * dot.orbitCenter.radius * growthFactor;
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                }
            } else {
                dot.angle += dot.radiusSpeed;
                const center = isDnaMode ? getRandomPosition() : { x: canvas.width / 2, y: canvas.height / 2 };
                if (dot.agent === 'agent1' && !isDnaMode) {
                    const radius = 10 * growthFactor;
                    dot.targetX = center.x + radius * Math.cos(dot.angle);
                    dot.targetY = center.y + radius * Math.sin(dot.angle) + (dot.index % 2 ? 10 : -10);
                } else if (dot.agent === 'agent2' && !isDnaMode) {
                    const cubeVertices = [
                        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                    ];
                    const vertex = cubeVertices[dot.index % cubeVertices.length];
                    const cubeSize = 15 * growthFactor;
                    const cosA = Math.cos(dot.angle);
                    const sinA = Math.sin(dot.angle);
                    const rotX = vertex[0] * cosA - vertex[1] * sinA;
                    const rotY = vertex[0] * sinA + vertex[1] * cosA;
                    const isoX = (rotX - vertex[2]) * cubeSize * 0.707;
                    const isoY = (rotX + vertex[2]) * cubeSize * 0.5;
                    dot.targetX = center.x + isoX;
                    dot.targetY = center.y + isoY;
                } else if (dot.agent === 'agent3' && !isDnaMode) {
                    const radius = 15 * growthFactor;
                    dot.targetX = center.x + radius * Math.cos(dot.angle);
                    dot.targetY = center.y + radius * Math.sin(dot.angle);
                } else if (dot.agent === 'agent4' && !isDnaMode) {
                    const starAngle = (dot.index % 5) * (2 * Math.PI / 5);
                    const offset = dot.index < 15 ? 0 : Math.PI / 5;
                    const starRadius = (dot.index < 15 ? 15 : 10) * growthFactor;
                    dot.targetX = center.x + starRadius * Math.cos(starAngle + dot.angle);
                    dot.targetY = center.y + starRadius * Math.sin(starAngle + dot.angle);
                } else if (isDnaMode) {
                    const dnaRadius = 20 * growthFactor;
                    const strandOffset = (dot.index % 4) * 5;
                    const phaseShift = (dot.agent === 'agent1' ? 0 : dot.agent === 'agent2' ? Math.PI / 2 : dot.agent === 'agent3' ? Math.PI : 3 * Math.PI / 2);
                    dot.targetX = center.x + dnaRadius * Math.cos(dot.angle + phaseShift) + strandOffset;
                    dot.targetY = center.y + dnaRadius * Math.sin(dot.angle + phaseShift);
                } else {
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -0.8;
                    if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -0.8;
                    dot.x = Math.max(0, Math.min(canvas.width, dot.x));
                    dot.y = Math.max(0, Math.min(canvas.height, dot.y));
                }
                if (!isGalaxyMode) {
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                }
            }
        });

        if (animationPhase === 'explode') {
            animationProgress += 0.02;
            if (animationProgress >= 1) {
                animationPhase = 'reform';
                animationProgress = 0;
                dots.forEach(dot => {
                    dot.opacity = 0;
                    dot.vx = 0;
                    dot.vy = 0;
                });
            }
        } else if (animationPhase === 'reform') {
            animationProgress += 0.01;
            dots.forEach(dot => {
                dot.opacity = animationProgress;
                dot.x += (dot.targetX - dot.x) * 0.1;
                dot.y += (dot.targetY - dot.y) * 0.1;
            });
            if (animationProgress >= 1) {
                animationPhase = 'none';
            }
        } else {
            if (isDnaMode || isGalaxyMode || activeAgents.length > 0) {
                growthFactor += 0.002;
            }
            if (isGalaxyMode && Math.random() < 0.02 && dotCount < 250) {
                const newIndex = dotCount++;
                const newDot = createDot('galaxy', newIndex, `agent${Math.floor(Math.random() * 4) + 1}`);
                if (newDot) dots.push(newDot);
            }
        }
    }

    // Animation loop
    function animate() {
        updateDots();
        drawDots();
        requestAnimationFrame(animate);
    }

    // Override console.clear to trigger explosion
    const originalConsoleClear = console.clear;
    console.clear = function () {
        originalConsoleClear.apply(console);
        animationPhase = 'explode';
        animationProgress = 0;
        dots.forEach(dot => {
            dot.vx = (Math.random() - 0.5) * 10;
            dot.vy = (Math.random() - 0.5) * 10;
            dot.opacity = 1;
        });
    };

    // Global function to set active agents
    window.setAgentsActive = function (active, agents = [], dnaMode = false, galaxyMode = false) {
        if (active && (agents.length !== activeAgents.length || agents.some((a, i) => a !== activeAgents[i]) || dnaMode !== isDnaMode || galaxyMode !== isGalaxyMode)) {
            activeAgents = agents;
            isDnaMode = dnaMode;
            isGalaxyMode = galaxyMode;
            initDots();
        } else if (!active && (activeAgents.length > 0 || isDnaMode || isGalaxyMode)) {
            activeAgents = [];
            isDnaMode = false;
            isGalaxyMode = false;
            initDots();
        }
    };

    // Initialize neural dots
    window.initNeurots = function () {
        if (!canvas || !ctx) {
            console.error('Canvas or context not found');
            return;
        }
        setCanvasSize();
        initDots();
        animate();
        window.addEventListener('resize', setCanvasSize);
    };
})();
