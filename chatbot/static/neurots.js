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
    let isRandomMode = false;
    let isMorphCollabMode = false;
    let isGalaxyMode = false;
    let growthFactor = 1;
    let dotCount = 30;
    const MAX_DISTANCE = 80;
    const MAX_COLLAB_DISTANCE = 120;
    let patternSide = 'center';
    let animationPhase = 'dissipate';
    let animationProgress = 0;
    let stopRequested = false;
    let isInitialLoad = true;

    function setCanvasSize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        console.log('Canvas resized:', canvas.width, canvas.height);
    }

    function getAgentColor(agent) {
        const colors = {
            'agent1': '#00cc00',
            'agent2': '#00ff99',
            'agent3': '#ff33cc',
            'agent4': '#33ccff'
        };
        return colors[agent] || '#00ff00';
    }

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
            if (isMobile) {
                x = Math.random() * canvas.width;
                y = Math.random() * 100;
            } else {
                x = Math.random() * canvas.width;
                y = Math.random() * canvas.height;
            }
        } while (
            x > textContainer.x && x < textContainer.x + textContainer.width &&
            y > textContainer.y && y < textContainer.y + textContainer.height
        );
        return { x, y };
    }

    function createDot(pattern, index, agent) {
        const base = {
            radius: Math.random() * 2 + 1,
            color: getAgentColor(agent),
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            angle: 0,
            radiusSpeed: 0,
            opacity: 1,
            targetX: 0,
            targetY: 0,
            index: index,
            agent: agent
        };
        let center = (isRandomMode || isMorphCollabMode) ? getRandomPosition() : { x: canvas.width / 2, y: canvas.height / 2 };
        if (isGalaxyMode) {
            center = { x: Math.random() * canvas.width, y: Math.random() * canvas.height };
            base.color = getAgentColor(index < 4 ? `agent${index + 1}` : `agent${Math.floor(Math.random() * 4) + 1}`);
            base.type = index < 4 ? 'star' : Math.random() < 0.5 ? 'star' : Math.random() < 0.7 ? 'planet' : Math.random() < 0.9 ? 'comet' : 'asteroid';
            base.opacity = base.type === 'star' ? Math.random() * 0.5 + 0.5 : 1;
            base.radius = base.type === 'planet' ? 3 + Math.random() * 2 : base.type === 'comet' ? 2 : 1;
            base.vx = base.type === 'comet' ? (Math.random() - 0.5) * 4 : base.vx;
            base.vy = base.type === 'comet' ? (Math.random() - 0.5) * 4 : base.vy;
            base.trail = base.type === 'comet' ? [] : null;
        }
        let dot;
        switch (pattern) {
            case 'helix':
                const helixAngle = index * 0.6;
                const helixRadius = (10 + index * 1) * growthFactor;
                dot = {
                    ...base,
                    x: center.x,
                    y: center.y,
                    targetX: center.x + helixRadius * Math.cos(helixAngle),
                    targetY: center.y + helixRadius * Math.sin(helixAngle) + (index % 2 ? 10 : -10),
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    angle: helixAngle,
                    radiusSpeed: 0.03
                };
                break;
            case 'cube':
                const cubeVertices = [
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                    [0, -1, -1], [0, 1, -1], [-1, 0, -1], [1, 0, -1],
                    [0, -1, 1], [0, 1, 1], [-1, 0, 1], [1, 0, 1]
                ];
                const vertex = cubeVertices[index % cubeVertices.length];
                const cubeSize = 15 * growthFactor;
                const isoX = (vertex[0] - vertex[1]) * cubeSize * 0.707;
                const isoY = (vertex[0] + vertex[1] - 2 * vertex[2]) * cubeSize * 0.5;
                dot = {
                    ...base,
                    x: center.x,
                    y: center.y,
                    targetX: center.x + isoX,
                    targetY: center.y + isoY,
                    vx: (Math.random() - 0.5) * 0.2,
                    vy: (Math.random() - 0.5) * 0.2,
                    angle: index * 0.4,
                    radiusSpeed: 0.02
                };
                break;
            case 'torus':
                const torusAngle = index * 0.4;
                const torusRadius = 15 * growthFactor;
                dot = {
                    ...base,
                    x: center.x,
                    y: center.y,
                    targetX: center.x + torusRadius * Math.cos(torusAngle),
                    targetY: center.y + torusRadius * Math.sin(torusAngle),
                    vx: (Math.random() - 0.5) * 0.3,
                    vy: (Math.random() - 0.5) * 0.3,
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
                    x: center.x,
                    y: center.y,
                    targetX: center.x + starRadius * Math.cos(starAngle + offset),
                    targetY: center.y + starRadius * Math.sin(starAngle + offset),
                    vx: (Math.random() - 0.5) * 0.3,
                    vy: (Math.random() - 0.5) * 0.3,
                    angle: starAngle,
                    radiusSpeed: 0.015
                };
                break;
            default:
                dot = {
                    ...base,
                    x: center.x,
                    y: center.y,
                    targetX: center.x + (Math.random() - 0.5) * canvas.width,
                    targetY: center.y + (Math.random() - 0.5) * canvas.height,
                    vx: (Math.random() - 0.5) * 2,
                    vy: (Math.random() - 0.5) * 2
                };
        }
        console.log(`Created dot for ${pattern} (agent: ${agent}):`, dot);
        return dot;
    }

    function initDots() {
        dots = [];
        if (isGalaxyMode) {
            // Start with one star per agent
            for (let i = 0; i < 4; i++) {
                const dot = createDot('galaxy', i, `agent${i + 1}`);
                if (dot) dots.push(dot);
            }
            dotCount = 4; // Start with 4 stars
        } else if (isMorphCollabMode) {
            activeAgents = ['agent1', 'agent2', 'agent3', 'agent4'];
            activeAgents.forEach(agent => {
                const pattern = { agent1: 'helix', agent2: 'cube', agent3: 'torus', agent4: 'star' }[agent];
                for (let i = 0; i < 30; i++) {
                    const dot = createDot(pattern, i, agent);
                    if (dot) dots.push(dot);
                }
            });
            dotCount = 30 * 4; // Fixed for morphcollab
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
            dotCount = 30; // Fixed for agents
        }
        console.log(`Initialized ${dots.length} dots for pattern: ${isGalaxyMode ? 'galaxy' : isMorphCollabMode ? 'morphcollab' : activeAgents[0] || 'random'}, randomMode: ${isRandomMode}, growthFactor: ${growthFactor}, dotCount: ${dotCount}`);
        if (isInitialLoad || !activeAgents.length) {
            animationPhase = 'dissipate';
        } else {
            animationPhase = 'reform';
        }
        animationProgress = 0;
        if (isRandomMode || isMorphCollabMode || isGalaxyMode) {
            growthFactor = 1;
            stopRequested = false;
        }
    }

    function checkBounds() {
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
            if (dot.type === 'comet' && dot.trail) {
                ctx.beginPath();
                ctx.moveTo(dot.x, dot.y);
                dot.trail.forEach(point => ctx.lineTo(point.x, point.y));
                ctx.strokeStyle = `rgba(${parseInt(dot.color.slice(1, 3), 16)}, ${parseInt(dot.color.slice(3, 5), 16)}, ${parseInt(dot.color.slice(5, 7), 16)}, 0.5)`;
                ctx.stroke();
            }
        });

        const maxDistance = isMorphCollabMode ? MAX_COLLAB_DISTANCE : MAX_DISTANCE;
        for (let i = 0; i < dots.length; i++) {
            for (let j = i + 1; j < dots.length; j++) {
                if (!dots[i] || !dots[j]) continue;
                if (isMorphCollabMode && dots[i].agent !== dots[j].agent) {
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
                } else if (!isMorphCollabMode) {
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

    function updateDots() {
        if (stopRequested || (isRandomMode && checkBounds())) {
            console.log('Stopping mode: ', { stopRequested, boundsHit: checkBounds() });
            isRandomMode = false;
            isMorphCollabMode = false;
            isGalaxyMode = false;
            growthFactor = 1;
            dotCount = 30;
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
                isInitialLoad = false;
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
                        if (dot.trail.length > 5) dot.trail.shift();
                    } else if (dot.type === 'asteroid') {
                        dot.x += dot.vx * 2;
                        dot.y += dot.vy * 2;
                        if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                        if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                    }
                } else if (activeAgents.length > 0) {
                    const center = (isRandomMode || isMorphCollabMode) ? getRandomPosition() : { x: canvas.width / 2, y: canvas.height / 2 };
                    if (dot.agent === 'agent1') {
                        dot.angle += dot.radiusSpeed;
                        const radius = (10 + dot.radius * 1) * growthFactor;
                        dot.targetX = center.x + radius * Math.cos(dot.angle);
                        dot.targetY = center.y + radius * Math.sin(dot.angle) + (dot.angle % 2 ? 10 : -10);
                        dot.x += (dot.targetX - dot.x) * 0.05;
                        dot.y += (dot.targetY - dot.y) * 0.05;
                    } else if (dot.agent === 'agent2') {
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
                        const rotX = vertex[0] * cosA - vertex[1] * sinA;
                        const rotY = vertex[0] * sinA + vertex[1] * cosA;
                        const isoX = (rotX - vertex[2]) * cubeSize * 0.707;
                        const isoY = (rotX + vertex[2]) * cubeSize * 0.5;
                        dot.targetX = center.x + isoX;
                        dot.targetY = center.y + isoY;
                        dot.x += (dot.targetX - dot.x) * 0.05;
                        dot.y += (dot.targetY - dot.y) * 0.05;
                        dot.vx = dot.vx || (Math.random() - 0.5) * 0.2;
                        dot.vy = dot.vy || (Math.random() - 0.5) * 0.2;
                        dot.x += dot.vx;
                        dot.y += dot.vy;
                        const bound = 15 * growthFactor;
                        if (Math.abs(dot.x - dot.targetX) > bound) dot.vx *= -0.9;
                        if (Math.abs(dot.y - dot.targetY) > bound) dot.vy *= -0.9;
                    } else if (dot.agent === 'agent3') {
                        dot.angle += dot.radiusSpeed;
                        const radius = 15 * growthFactor;
                        dot.targetX = center.x + radius * Math.cos(dot.angle);
                        dot.targetY = center.y + radius * Math.sin(dot.angle);
                        dot.x += (dot.targetX - dot.x) * 0.05;
                        dot.y += (dot.targetY - dot.y) * 0.05;
                    } else if (dot.agent === 'agent4') {
                        dot.angle += dot.radiusSpeed;
                        const starAngle = (dot.index % 5) * (2 * Math.PI / 5);
                        const offset = dot.index < 15 ? 0 : Math.PI / 5;
                        const starRadius = (dot.index < 15 ? 15 : 10) * growthFactor;
                        dot.targetX = center.x + starRadius * Math.cos(starAngle + offset + dot.angle);
                        dot.targetY = center.y + starRadius * Math.sin(starAngle + offset + dot.angle);
                        dot.x += (dot.targetX - dot.x) * 0.05;
                        dot.y += (dot.targetY - dot.y) * 0.05;
                        dot.vx = dot.vx || (Math.random() - 0.5) * 0.3;
                        dot.vy = dot.vy || (Math.random() - 0.5) * 0.3;
                        dot.x += dot.vx;
                        dot.y += dot.vy;
                        const bound = 15 * growthFactor;
                        if (Math.abs(dot.x - center.x) > bound) dot.vx *= -0.9;
                        if (Math.abs(dot.y - center.y) > bound) dot.vy *= -0.9;
                    }
                    if (isMorphCollabMode) {
                        const morphCenterX = canvas.width / 2;
                        const morphCenterY = canvas.height / 2;
                        dot.targetX = dot.targetX + (morphCenterX - dot.targetX) * 0.005;
                        dot.targetY = dot.targetY + (morphCenterY - dot.targetY) * 0.005;
                    }
                } else {
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                    if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                }
            });
            if (isRandomMode) {
                growthFactor += 0.005;
            } else if (isMorphCollabMode) {
                growthFactor += 0.002;
            } else if (isGalaxyMode) {
                growthFactor += 0.002;
                if (Math.random() < 0.02 && dotCount < 200) { // Slowly add dots
                    const newIndex = dotCount++;
                    const newDot = createDot('galaxy', newIndex, `agent${Math.floor(Math.random() * 4) + 1}`);
                    if (newDot) dots.push(newDot);
                }
            }
        }
    }

    function animate() {
        updateDots();
        drawDots();
        requestAnimationFrame(animate);
    }

    window.setAgentsActive = function (active, agents = [], randomMode = false, morphCollab = false, galaxyMode = false) {
        if (active && (agents.length !== activeAgents.length || agents.some((a, i) => a !== activeAgents[i]) || randomMode !== isRandomMode || morphCollab !== isMorphCollabMode || galaxyMode !== isGalaxyMode)) {
            activeAgents = agents;
            isRandomMode = randomMode;
            isMorphCollabMode = morphCollab;
            isGalaxyMode = galaxyMode;
            stopRequested = false;
            patternSide = 'center';
            console.log(`setAgentsActive: agents=${agents.join(',')}, randomMode=${randomMode}, morphCollab=${morphCollab}, galaxyMode=${galaxyMode}, patternSide=${patternSide}`);
            initDots();
        } else if (!active && (activeAgents.length > 0 || isRandomMode || isMorphCollabMode || isGalaxyMode)) {
            activeAgents = [];
            isRandomMode = false;
            isMorphCollabMode = false;
            isGalaxyMode = false;
            stopRequested = false;
            patternSide = 'center';
            console.log('Resetting to default pattern');
            initDots();
        }
    };

    window.stopRandomMode = function () {
        stopRequested = true;
        isRandomMode = false;
        isMorphCollabMode = false;
        isGalaxyMode = false;
        growthFactor = 1;
        dotCount = 30;
        console.log('Random mode stopped');
        initDots();
    };

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
