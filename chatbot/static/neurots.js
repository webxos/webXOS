function setNeuralPattern(agent) {
    try {
        const canvas = document.getElementById('neuralCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const patterns = {
            agent1: { color: '#00ff00', shape: 'circle' },
            agent2: { color: '#00ffff', shape: 'square' },
            agent3: { color: '#ff00ff', shape: 'triangle' },
            agent4: { color: '#0000ff', shape: 'star' },
            agentic: [
                { color: '#00ff00', shape: 'circle' },
                { color: '#00ffff', shape: 'square' },
                { color: '#ff00ff', shape: 'triangle' },
                { color: '#0000ff', shape: 'star' }
            ]
        };

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!agent) return;

        const drawShape = (x, y, type, color, size) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            if (type === 'circle') {
                ctx.arc(x, y, size, 0, Math.PI * 2);
            } else if (type === 'square') {
                ctx.rect(x - size, y - size, size * 2, size * 2);
            } else if (type === 'triangle') {
                ctx.moveTo(x, y - size);
                ctx.lineTo(x - size, y + size);
                ctx.lineTo(x + size, y + size);
                ctx.closePath();
            } else if (type === 'star') {
                for (let i = 0; i < 5; i++) {
                    ctx.lineTo(
                        x + size * Math.cos((Math.PI * 2 * i) / 5 - Math.PI / 2),
                        y + size * Math.sin((Math.PI * 2 * i) / 5 - Math.PI / 2)
                    );
                    ctx.lineTo(
                        x + (size / 2) * Math.cos((Math.PI * 2 * i + Math.PI) / 5 - Math.PI / 2),
                        y + (size / 2) * Math.sin((Math.PI * 2 * i + Math.PI) / 5 - Math.PI / 2)
                    );
                }
                ctx.closePath();
            }
            ctx.fill();
        };

        const selected = patterns[agent] || patterns.agentic;
        const dots = Array.isArray(selected) ? selected : [selected];
        dots.forEach(({ color, shape }) => {
            for (let i = 0; i < 10; i++) {
                drawShape(
                    Math.random() * canvas.width,
                    Math.random() * canvas.height,
                    shape,
                    color,
                    10 + Math.random() * 10
                );
            }
        });
    } catch (error) {
        console.error('Neural visualization failed:', error);
    }
}

window.setNeuralPattern = setNeuralPattern;
