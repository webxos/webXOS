const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
let dots = [];
let activeAgent = null;
const DOT_COUNT = 50;
const MAX_DISTANCE = 100;

function setCanvasSize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function createDot() {
    return {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        radius: Math.random() * 2 + 1,
        color: activeAgent ? getAgentColor(activeAgent) : '#00ff00'
    };
}

function getAgentColor(agent) {
    const colors = {
        'agent1': '#00cc00',
        'agent2': '#00ff99',
        'agent3': '#33ff00',
        'agent4': '#66ff00'
    };
    return colors[agent] || '#00ff00';
}

function initDots() {
    dots = [];
    for (let i = 0; i < DOT_COUNT; i++) {
        dots.push(createDot());
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
                ctx.strokeStyle = `rgba(0, 255, 0, ${1 - distance / MAX_DISTANCE})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
}

function updateDots() {
    dots.forEach(dot => {
        dot.x += dot.vx;
        dot.y += dot.vy;

        if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
        if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
    });
}

function animate() {
    updateDots();
    drawDots();
    requestAnimationFrame(animate);
}

window.setAgentsActive = function(active, agent = null) {
    activeAgent = active ? agent : null;
    dots.forEach(dot => {
        dot.color = activeAgent ? getAgentColor(activeAgent) : '#00ff00';
    });
};

window.initNeurots = function() {
    setCanvasSize();
    initDots();
    animate();
    window.addEventListener('resize', setCanvasSize);
};
