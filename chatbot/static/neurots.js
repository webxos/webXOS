const initNeurots = () => {
    const canvas = document.getElementById('neuralCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    let dots = [];
    const numDots = 50;
    let explosionTime = 0;
    let activePattern = null;
    let scale = 0.1;
    let rotation = 0;

    // Initialize dots for explosion
    const initDots = () => {
        dots = [];
        for (let i = 0; i < numDots; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 5 + 2;
            dots.push({
                x: canvas.width / 2,
                y: canvas.height / 2,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                radius: Math.random() * 2 + 1,
                alpha: 1
            });
        }
    };

    // Draw background dots
    const drawDots = () => {
        if (explosionTime < 60) {
            dots.forEach(dot => {
                dot.x += dot.vx;
                dot.y += dot.vy;
                dot.alpha = 1 - explosionTime / 60;
                ctx.globalAlpha = dot.alpha;
                ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
                ctx.beginPath();
                ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
                ctx.fill();
            });
            explosionTime++;
        } else {
            dots.forEach(dot => {
                dot.x += dot.vx;
                dot.y += dot.vy;
                if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                ctx.globalAlpha = 0.5;
                ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
                ctx.beginPath();
                ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
                ctx.fill();
            });
            dots.forEach((dot, i) => {
                dots.slice(i + 1).forEach(otherDot => {
                    const dx = dot.x - otherDot.x;
                    const dy = dot.y - otherDot.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < 100) {
                        ctx.globalAlpha = 1 - distance / 100;
                        ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
                        ctx.beginPath();
                        ctx.moveTo(dot.x, dot.y);
                        ctx.lineTo(otherDot.x, otherDot.y);
                        ctx.stroke();
                    }
                });
            });
        }
    };

    // Agent-specific patterns
    const drawHelix = () => {
        ctx.strokeStyle = '#00cc00';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        rotation += 0.05;
        ctx.beginPath();
        for (let t = -Math.PI; t <= Math.PI; t += 0.1) {
            const x1 = centerX + Math.cos(t + rotation) * scale * 50;
            const y1 = centerY + t * 50;
            const x2 = centerX + Math.cos(t + rotation + Math.PI) * scale * 50;
            const y2 = centerY + t * 50;
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
        }
        ctx.stroke();
        scale = Math.min(scale + 0.01, 1);
    };

    const drawCube = () => {
        ctx.strokeStyle = '#00ff99';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const size = scale * 100;
        rotation += 0.05;
        ctx.beginPath();
        const vertices = [
            [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
            [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
        ];
        const edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ];
        const cos = Math.cos(rotation);
        const sin = Math.sin(rotation);
        vertices.forEach(v => {
            const x = v[0] * cos - v[2] * sin;
            const z = v[0] * sin + v[2] * cos;
            v[0] = x;
            v[2] = z;
        });
        edges.forEach(([i, j]) => {
            ctx.moveTo(centerX + vertices[i][0], centerY + vertices[i][1]);
            ctx.lineTo(centerX + vertices[j][0], centerY + vertices[j][1]);
        });
        ctx.stroke();
        scale = Math.min(scale + 0.01, 1);
    };

    const drawTorus = () => {
        ctx.strokeStyle = '#ff33cc';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        rotation += 0.05;
        ctx.beginPath();
        for (let theta = 0; theta < Math.PI * 2; theta += 0.1) {
            for (let phi = 0; phi < Math.PI * 2; phi += 0.1) {
                const x = (100 * scale + 30 * Math.cos(phi)) * Math.cos(theta);
                const y = (100 * scale + 30 * Math.cos(phi)) * Math.sin(theta);
                const z = 30 * Math.sin(phi);
                const x2 = x * Math.cos(rotation) - z * Math.sin(rotation);
                const y2 = y;
                ctx.lineTo(centerX + x2, centerY + y2);
            }
        }
        ctx.stroke();
        scale = Math.min(scale + 0.01, 1);
    };

    const drawStar = () => {
        ctx.strokeStyle = '#33ccff';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
            const outerX = centerX + Math.cos(i * 4 * Math.PI / 5 + rotation) * scale * 100;
            const outerY = centerY + Math.sin(i * 4 * Math.PI / 5 + rotation) * scale * 100;
            const innerX = centerX + Math.cos((i + 0.5) * 4 * Math.PI / 5 + rotation) * scale * 50;
            const innerY = centerY + Math.sin((i + 0.5) * 4 * Math.PI / 5 + rotation) * scale * 50;
            ctx.lineTo(outerX, outerY);
            ctx.lineTo(innerX, innerY);
        }
        ctx.closePath();
        ctx.stroke();
        scale = Math.min(scale + 0.01, 1);
        rotation += 0.05;
    };

    const drawDNA = () => {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        rotation += 0.05;
        ctx.beginPath();
        for (let t = -Math.PI; t <= Math.PI; t += 0.1) {
            const x1 = centerX + Math.cos(t + rotation) * scale * 50;
            const y1 = centerY + t * 50;
            const x2 = centerX + Math.cos(t + rotation + Math.PI) * scale * 50;
            const y2 = centerY + t * 50;
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
        }
        ctx.stroke();
        scale = Math.min(scale + 0.01, 1);
    };

    const drawGalaxy = () => {
        ctx.strokeStyle = '#33ccff';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        ctx.beginPath();
        for (let t = 0; t < Math.PI * 6; t += 0.1) {
            const r = scale * 30 * Math.exp(t / 6);
            const x = centerX + r * Math.cos(t + rotation);
            const y = centerY + r * Math.sin(t + rotation);
            ctx.lineTo(x, y);
        }
        ctx.stroke();
        scale = Math.min(scale + 0.01, 1);
        rotation += 0.05;
    };

    const animate = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawDots();
        if (activePattern) {
            activePattern();
        }
        requestAnimationFrame(animate);
    };

    // Public API
    window.setAgentsActive = (active, agents = [], dna = false, galaxy = false) => {
        scale = 0.1;
        rotation = 0;
        if (!active) {
            activePattern = null;
            return;
        }
        if (galaxy) {
            activePattern = drawGalaxy;
        } else if (dna) {
            activePattern = drawDNA;
        } else if (agents.length === 1) {
            const agent = agents[0];
            if (agent === 'vial1') activePattern = drawHelix;
            else if (agent === 'vial2') activePattern = drawCube;
            else if (agent === 'vial3') activePattern = drawTorus;
            else if (agent === 'vial4') activePattern = drawStar;
        }
    };

    window.initNeurots = () => {
        initDots();
        animate();
    };

    // Trigger explosion on init
    initDots();
    animate();
};
