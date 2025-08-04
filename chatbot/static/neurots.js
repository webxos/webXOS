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
    let animationPhase = 'dissipate';
    let animationProgress = 0;

    // Set canvas size
    function setCanvasSize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        console.log('Canvas resized:', canvas.width, canvas.height);
    }

    // Get agent color
    function getAgentColor(agent) {
        const colors = {
            'agent1': '#00cc00',
            'agent2': '#00ff99',
            'agent3': '#ff33cc',
            'agent4': '#33ccff'
        };
        return colors[agent] || '#00ff00';
    }

    // Get position avoiding chatbox for dna/galaxy
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
    function createDot(pattern, index, agent, strandIndex = 0) {
        const base = {
            radius: 2,
            color: getAgentColor(agent),
            angle: 0,
            radiusSpeed: 0,
            opacity: 1,
            targetX: 0,
            targetY: 0,
            index: index,
            agent: agent,
            x: canvas.width / 2,
            y: canvas.height / 2,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2
        };
        let center = isDnaMode || isGalaxyMode ? getRandomPosition() : { x: canvas.width / 2, y: canvas.height / 2 };
        let dot;
        switch (pattern) {
            case 'globe':
                const phi = Math.acos(1 - 2 * (index + 0.5) / 30);
                const theta = Math.PI * (3 - Math.sqrt(5)) * index;
                const globeRadius = 15 * growthFactor;
                dot = {
                    ...base,
                    targetX: center.x + globeRadius * Math.sin(phi) * Math.cos(theta),
                    targetY: center.y + globeRadius * Math.sin(phi) * Math.sin(theta),
                    angle: index * 0.2,
                    radiusSpeed: 0.02
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
                const dnaRadius = (strandIndex % 2 === 0 ? 15 : 30) * growthFactor;
                const dnaAngle = index * 0.2;
                const strandOffset = (index % 2) ? 10 : -10;
                dot = {
                    ...base,
                    targetX: center.x + dnaRadius * Math.cos(dnaAngle) + strandOffset,
                    targetY: center.y + dnaRadius * Math.sin(dnaAngle),
                    angle: dnaAngle,
                    radiusSpeed: 0.01,
                    strandIndex: strandIndex
                };
                break;
            case 'galaxy':
                base.color = getAgentColor(index < 4 ? `agent${index + 1}` : `agent${Math.floor(Math.random() * 4) + 1}`);
                base.type = index < 4 ? 'star' : Math.random() < 0.3 ? 'star' : Math.random() < 0.6 ? 'planet' : Math.random() < 0.9 ? 'comet' : 'smallPlanet';
                base.opacity = base.type === 'star' ? Math.random() * 0.5 + 0.5 : 1;
                base.radius = base.type === 'planet' ? 3 + Math.random() * 2 : base.type === 'comet' ? 2 : base.type === 'smallPlanet' ? 1 + Math.random() * 1 : 0.5 + Math.random() * 0.5;
                base.vx = base.type === 'comet' ? (Math.random() - 0.5) * 6 : (Math.random() - 0.5) * 2;
                base.vy = base.type === 'comet' ? (Math.random() - 0.5) * 6 : (Math.random() - 0.5) * 2;
                base.trail = base.type === 'comet' ? [] : null;
                base.orbitCenter = base.type === 'smallPlanet' ? { x: center.x, y: center.y, radius: 10 + Math.random() * 10 } : null;
                base.glow = base.type === 'star' ? Math.random() * 2 + 1 : 0;
                dot = { ...base };
                break;
            default:
                dot = {
                    ...base,
                    targetX: center.x + (Math.random() - 0.5) * 50,
                    targetY: center.y + (Math.random() - 0.5) * 50
                };
        }
        console.log(`Created dot for ${pattern} (agent: ${agent}, strand: ${strandIndex}):`, dot);
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
            const strandCount = Math.floor(Math.random() * 6) + 5; // 5-10 strands
            const colorPairs = [
                ['agent1', 'agent2'],
                ['agent2', 'agent3'],
                ['agent3', 'agent4'],
                ['agent4', 'agent1'],
                ['agent1', 'agent3'],
                ['agent2', 'agent4']
            ];
            for (let s = 0; s < strandCount; s++) {
                const pair = colorPairs[s % colorPairs.length];
                const pattern = 'dna';
                for (let i = 0; i < 20; i++) {
                    const dot1 = createDot(pattern, i, pair[0], s);
                    const dot2 = createDot(pattern, i, pair[1], s);
                    if (dot1) dots.push(dot1);
                    if (dot2) dots.push(dot2);
                }
            }
            dotCount = strandCount * 20 * 2;
        } else {
            const pattern = activeAgents.length > 0 ? {
                'agent1': 'globe',
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
        console.log(`Initialized ${dots.length} dots for pattern: ${isGalaxyMode ? 'galaxy' : isDnaMode ? 'dna' : activeAgents[0] || 'random'}, growthFactor: ${growthFactor}, dotCount: ${dotCount}`);
        animationPhase = 'dissipate';
        animationProgress = 0;
        growthFactor = 0.5;
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
                if (isDnaMode && dots[i].agent !== dots[j].agent && dots[i].strandIndex === dots[j].strandIndex) {
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
                } else if (!isDnaMode && !isGalaxyMode && animationPhase !== 'dissipate') {
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
            if (isGalaxyMode) {
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
            } else if (animationPhase === 'dissipate') {
                dot.x += dot.vx;
                dot.y += dot.vy;
                if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                dot.opacity = Math.max(0, 1 - animationProgress);
            } else {
                dot.angle += dot.radiusSpeed;
                const center = isDnaMode ? getRandomPosition() : { x: canvas.width / 2, y: canvas.height / 2 };
                if (dot.agent === 'agent1' && !isDnaMode) {
                    const phi = Math.acos(1 - 2 * (dot.index + 0.5) / 30);
                    const theta = Math.PI * (3 - Math.sqrt(5)) * dot.index + dot.angle;
                    const globeRadius = 15 * growthFactor;
                    dot.targetX = center.x + globeRadius * Math.sin(phi) * Math.cos(theta);
                    dot.targetY = center.y + globeRadius * Math.sin(phi) * Math.sin(theta);
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
                    const starRadius = (index < 15 ? 15 : 10) * growthFactor;
                    dot.targetX = center.x + starRadius * Math.cos(starAngle + dot.angle);
                    dot.targetY = center.y + starRadius * Math.sin(starAngle + dot.angle);
                } else if (isDnaMode) {
                    const dnaRadius = (dot.strandIndex % 2 === 0 ? 15 : 30) * growthFactor;
                    const strandOffset = (dot.index % 2) ? 10 : -10;
                    dot.targetX = center.x + dnaRadius * Math.cos(dot.angle) + strandOffset;
                    dot.targetY = center.y + dnaRadius * Math.sin(dot.angle);
                } else {
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                    if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                }
                if (!isGalaxyMode && animationPhase !== 'dissipate') {
                    dot.x += (dot.targetX - dot.x) * 0.05;
                    dot.y += (dot.targetY - dot.y) * 0.05;
                }
            }
        });

        if (animationPhase === 'dissipate') {
            animationProgress += 0.02;
            if (animationProgress >= 1) {
                animationPhase = 'bounce';
            }
        } else if (animationPhase === 'bounce') {
            // Dots continue bouncing, no auto-reset
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

    // Global function to set active agents
    window.setAgentsActive = function (active, agents = [], dnaMode = false, galaxyMode = false) {
        if (active && (agents.length !== activeAgents.length || agents.some((a, i) => a !== activeAgents[i]) || dnaMode !== isDnaMode || galaxyMode !== isGalaxyMode)) {
            activeAgents = agents;
            isDnaMode = dnaMode;
            isGalaxyMode = galaxyMode;
            console.log(`setAgentsActive: agents=${agents.join(',')}, dnaMode=${dnaMode}, galaxyMode=${galaxyMode}`);
            initDots();
        } else if (!active && (activeAgents.length > 0 || isDnaMode || isGalaxyMode)) {
            activeAgents = [];
            isDnaMode = false;
            isGalaxyMode = false;
            console.log('Resetting to default pattern');
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
        console.log('Neurots initialized');
    };
})();
