let dotCount = 30;
const MAX_DISTANCE = 80;
const MAX_COLLAB_DISTANCE = 120;
    let animationPhase = 'dissipate';
    let animationPhase = 'explode';
let animationProgress = 0;
let isInitialLoad = true;

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
            'agent1': '#00cc00', // Green
            'agent2': '#00ff99', // Cyan
            'agent3': '#ff33cc', // Pink
            'agent4': '#33ccff'  // Blue
};
return colors[agent] || '#00ff00';
}

    // Get position avoiding chatbox for dna/galaxy
    // Get random position avoiding chatbox
function getRandomPosition() {
const isMobile = window.innerWidth <= 768;
const textContainer = {
@@ -65,15 +64,17 @@
const base = {
radius: 2,
color: getAgentColor(agent),
            angle: 0,
            angle: Math.random() * Math.PI * 2,
radiusSpeed: 0,
opacity: 1,
targetX: 0,
targetY: 0,
index: index,
agent: agent,
x: canvas.width / 2,
            y: canvas.height / 2
            y: canvas.height / 2,
            vx: (Math.random() - 0.5) * 10, // Initial velocity for explosion
            vy: (Math.random() - 0.5) * 10
};
let center = isDnaMode || isGalaxyMode ? getRandomPosition() : { x: canvas.width / 2, y: canvas.height / 2 };
if (isGalaxyMode) {
@@ -141,27 +142,27 @@
};
break;
case 'dna':
                const dnaAngle = index * 0.3;
                const dnaRadius = 15 * growthFactor;
                const strandOffset = (index % 2) ? 10 : -10;
                const dnaAngle = index * 0.2; // Slower rotation
                const dnaRadius = 20 * growthFactor;
                const strandOffset = (index % 4) * 5; // Four interwoven strands
                const phaseShift = (agent === 'agent1' ? 0 : agent === 'agent2' ? Math.PI / 2 : agent === 'agent3' ? Math.PI : 3 * Math.PI / 2);
dot = {
...base,
                    targetX: center.x + dnaRadius * Math.cos(dnaAngle) + strandOffset,
                    targetY: center.y + dnaRadius * Math.sin(dnaAngle),
                    targetX: center.x + dnaRadius * Math.cos(dnaAngle + phaseShift) + strandOffset,
                    targetY: center.y + dnaRadius * Math.sin(dnaAngle + phaseShift),
angle: dnaAngle,
                    radiusSpeed: 0.01
                    radiusSpeed: 0.005 // Much slower for smooth DNA spirals
};
break;
default:
dot = {
...base,
                    targetX: center.x + (Math.random() - 0.5) * 50,
                    targetY: center.y + (Math.random() - 0.5) * 50,
                    vx: (Math.random() - 0.5) * 2,
                    vy: (Math.random() - 0.5) * 2
                    targetX: center.x + (Math.random() - 0.5) * canvas.width,
                    targetY: center.y + (Math.random() - 0.5) * canvas.height,
                    vx: (Math.random() - 0.5) * 8,
                    vy: (Math.random() - 0.5) * 8
};
}
        console.log(`Created dot for ${pattern} (agent: ${agent}):`, dot);
return dot;
}

@@ -197,8 +198,7 @@
}
dotCount = 30;
}
        console.log(`Initialized ${dots.length} dots for pattern: ${isGalaxyMode ? 'galaxy' : isDnaMode ? 'dna' : activeAgents[0] || 'random'}, growthFactor: ${growthFactor}, dotCount: ${dotCount}`);
        animationPhase = 'dissipate';
        animationPhase = isInitialLoad ? 'explode' : 'reform';
animationProgress = 0;
growthFactor = 0.5;
isInitialLoad = false;
@@ -265,7 +265,20 @@
function updateDots() {
dots.forEach(dot => {
if (!dot) return;
            if (isGalaxyMode) {
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
@@ -321,15 +334,18 @@
dot.targetX = center.x + starRadius * Math.cos(starAngle + dot.angle);
dot.targetY = center.y + starRadius * Math.sin(starAngle + dot.angle);
} else if (isDnaMode) {
                    const dnaRadius = 15 * growthFactor;
                    const strandOffset = (dot.index % 2) ? 10 : -10;
                    dot.targetX = center.x + dnaRadius * Math.cos(dot.angle) + strandOffset;
                    dot.targetY = center.y + dnaRadius * Math.sin(dot.angle);
                    const dnaRadius = 20 * growthFactor;
                    const strandOffset = (dot.index % 4) * 5;
                    const phaseShift = (dot.agent === 'agent1' ? 0 : dot.agent === 'agent2' ? Math.PI / 2 : dot.agent === 'agent3' ? Math.PI : 3 * Math.PI / 2);
                    dot.targetX = center.x + dnaRadius * Math.cos(dot.angle + phaseShift) + strandOffset;
                    dot.targetY = center.y + dnaRadius * Math.sin(dot.angle + phaseShift);
} else {
dot.x += dot.vx;
dot.y += dot.vy;
                    if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                    if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                    if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -0.8;
                    if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -0.8;
                    dot.x = Math.max(0, Math.min(canvas.width, dot.x));
                    dot.y = Math.max(0, Math.min(canvas.height, dot.y));
}
if (!isGalaxyMode) {
dot.x += (dot.targetX - dot.x) * 0.05;
@@ -338,22 +354,20 @@
}
});

        if (animationPhase === 'dissipate') {
        if (animationPhase === 'explode') {
animationProgress += 0.02;
            dots.forEach(dot => {
                if (!dot) return;
                dot.opacity = Math.max(0, 1 - animationProgress);
                dot.x += (Math.random() - 0.5) * 5;
                dot.y += (Math.random() - 0.5) * 5;
            });
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
                if (!dot) return;
dot.opacity = animationProgress;
dot.x += (dot.targetX - dot.x) * 0.1;
dot.y += (dot.targetY - dot.y) * 0.1;
@@ -380,19 +394,30 @@
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
@@ -407,6 +432,5 @@
initDots();
animate();
window.addEventListener('resize', setCanvasSize);
        console.log('Neurots initialized');
};
})();
