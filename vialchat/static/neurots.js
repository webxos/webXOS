const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
let dots = [];
let active = false;
let agents = [];
let isSync = false;
let isTrain = false;

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function createDot(x, y, color, radius) {
    return { x, y, color, radius, vx: Math.random() * 2 - 1, vy: Math.random() * 2 - 1, t: 0 };
}

function initNeurots() {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    for (let i = 0; i < 100; i++) {
        dots.push(createDot(Math.random() * canvas.width, Math.random() * canvas.height, '#00ff00', 2));
    }
    animate();
}

function setAgentsActive(isActive, agentNames = [], syncActive = false, trainActive = false) {
    active = isActive;
    agents = agentNames;
    isSync = syncActive;
    isTrain = trainActive;
    dots = [];
    if (!active) return;
    const colors = {
        agent1: '#00cc00',
        agent2: '#00ff99',
        agent3: '#ff33cc',
        agent4: '#33ccff'
    };
    if (isTrain) {
        for (let i = 0; i < 200; i++) {
            const hue = `hsl(${Math.random() * 30 + 50}, 100%, ${Math.random() * 20 + 70}%)`; // Yellow-white-gold hues
            dots.push(createDot(canvas.width / 2, canvas.height / 2, hue, 3));
        }
    } else {
        for (let i = 0; i < 100; i++) {
            const color = agents.length === 1 ? colors[agents[0]] : '#00ff00';
            dots.push(createDot(canvas.width / 2, canvas.height / 2, color, 2));
        }
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!active) {
        requestAnimationFrame(animate);
        return;
    }

    dots.forEach(dot => {
        dot.t += 0.05;
        if (isTrain) {
            const angle = dot.t * 2;
            const radius = 100 + Math.sin(dot.t) * 50;
            dot.x = canvas.width / 2 + Math.cos(angle) * radius;
            dot.y = canvas.height / 2 + Math.sin(angle) * radius;
            dot.radius = 3 + Math.sin(dot.t * 2) * 2; // Twinkling effect
        } else if (isSync) {
            const radius = 50;
            const angle = dot.t + (dot.y / canvas.height) * Math.PI * 2;
            dot.x = canvas.width / 2 + Math.cos(angle) * radius;
            dot.y += Math.sin(angle) * 0.5;
            if (dot.y > canvas.height) dot.y -= canvas.height;
        } else if (agents.length === 1) {
            const pattern = agents[0] === 'agent1' ? 'helix' : agents[0] === 'agent2' ? 'cube' : agents[0] === 'agent3' ? 'torus' : 'star';
            if (pattern === 'helix') {
                const angle = dot.t + (dot.y / canvas.height) * Math.PI * 2;
                dot.x = canvas.width / 2 + Math.cos(angle) * 50;
                dot.y += 1;
                if (dot.y > canvas.height) dot.y -= canvas.height;
            } else if (pattern === 'cube') {
                dot.x = canvas.width / 2 + Math.cos(dot.t) * 50;
                dot.y = canvas.height / 2 + Math.sin(dot.t) * 50;
            } else if (pattern === 'torus') {
                const angle = dot.t * 2;
                dot.x = canvas.width / 2 + Math.cos(angle) * (50 + Math.sin(angle * 2) * 20);
                dot.y = canvas.height / 2 + Math.sin(angle) * (50 + Math.cos(angle * 2) * 20);
            } else if (pattern === 'star') {
                const angle = dot.t * 3;
                dot.x = canvas.width / 2 + Math.cos(angle) * (50 + Math.sin(angle * 5) * 20);
                dot.y = canvas.height / 2 + Math.sin(angle) * (50 + Math.cos(angle * 5) * 20);
            }
        }

        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
        ctx.fillStyle = isTrain ? `${dot.color}ee` : dot.color;
        ctx.shadowBlur = isTrain ? 15 : 5;
        ctx.shadowColor = isTrain ? dot.color : dot.color;
        ctx.fill();
    });

    dots.forEach((dot, i) => {
        for (let j = i + 1; j < dots.length; j++) {
            const other = dots[j];
            const dist = Math.hypot(dot.x - other.x, dot.y - other.y);
            if (dist < 100) {
                ctx.beginPath();
                ctx.moveTo(dot.x, dot.y);
                ctx.lineTo(other.x, other.y);
                ctx.strokeStyle = isTrain ? `rgba(255, 255, 200, ${1 - dist / 100})` : `rgba(0, 255, 0, ${1 - dist / 100})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    });

    requestAnimationFrame(animate);
}

window.initNeurots = initNeurots;
window.setAgentsActive = setAgentsActive;
