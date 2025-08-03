let dots = [];
let animationFrameId;

function initNeurots() {
    const canvas = document.getElementById('neuralCanvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext('2d');

    function createDot(agent, type) {
        const colors = {
            'agent1': '#00cc00',
            'agent2': '#00ff99',
            'agent3': '#ff33cc',
            'agent4': '#33ccff'
        };
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            color: colors[agent] || '#00ff00',
            type: type
        };
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        dots.forEach(dot => {
            dot.x += dot.vx;
            dot.y += dot.vy;
            if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
            if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
            ctx.beginPath();
            ctx.arc(dot.x, dot.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = dot.color;
            ctx.fill();
        });
        animationFrameId = requestAnimationFrame(animate);
    }

    function startMcpVisualizer() {
        const leftCanvas = document.getElementById('mcpLeftCanvas');
        const rightCanvas = document.getElementById('mcpRightCanvas');
        leftCanvas.width = 100;
        leftCanvas.height = document.querySelector('.popup-content').offsetHeight;
        rightCanvas.width = 100;
        rightCanvas.height = document.querySelector('.popup-content').offsetHeight;
        const leftCtx = leftCanvas.getContext('2d');
        const rightCtx = rightCanvas.getContext('2d');
        dots = [];

        ['agent1', 'agent2', 'agent3', 'agent4'].forEach(agent => {
            for (let i = 0; i < 10; i++) {
                dots.push(createDot(agent, 'mcp'));
            }
        });

        function animateMcp() {
            leftCtx.clearRect(0, 0, leftCanvas.width, leftCanvas.height);
            rightCtx.clearRect(0, 0, rightCanvas.width, rightCanvas.height);
            dots.forEach(dot => {
                dot.x += dot.vx * 0.5;
                dot.y += dot.vy * 0.5;
                if (dot.x < 0 || dot.x > 100) dot.vx *= -1;
                if (dot.y < 0 || dot.y > leftCanvas.height) dot.vy *= -1;
                const ctx = Math.random() > 0.5 ? leftCtx : rightCtx;
                ctx.beginPath();
                ctx.arc(dot.x, dot.y, 3, 0, Math.PI * 2);
                ctx.fillStyle = dot.color;
                ctx.fill();
            });
            animationFrameId = requestAnimationFrame(animateMcp);
        }

        animateMcp();
    }

    function stopMcpVisualizer() {
        cancelAnimationFrame(animationFrameId);
        dots = [];
        const leftCanvas = document.getElementById('mcpLeftCanvas');
        const rightCanvas = document.getElementById('mcpRightCanvas');
        leftCanvas.getContext('2d').clearRect(0, 0, leftCanvas.width, leftCanvas.height);
        rightCanvas.getContext('2d').clearRect(0, 0, rightCanvas.width, rightCanvas.height);
    }

    window.startMcpVisualizer = startMcpVisualizer;
    window.stopMcpVisualizer = stopMcpVisualizer;

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });

    window.updateNeuralCanvas = function(agent, task, data) {
        dots.push(createDot(agent, task));
        if (dots.length > 50) dots.shift();
    };

    animate();
}

window.initNeurots = initNeurots;
