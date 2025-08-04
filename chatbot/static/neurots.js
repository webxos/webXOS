const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const dots = [];
const colors = {
  agent1: '#00cc00',
  agent2: '#00ffff',
  agent3: '#ff00ff',
  agent4: '#0000ff',
  agentic: ['#00cc00', '#00ffff', '#ff00ff', '#0000ff']
};
const shapes = ['circle', 'square', 'triangle', 'star'];

class Dot {
  constructor(x, y, color, shape, agent) {
    this.x = x;
    this.y = y;
    this.color = color;
    this.shape = shape;
    this.agent = agent;
    this.vx = (Math.random() - 0.5) * 2;
    this.vy = (Math.random() - 0.5) * 2;
  }

  draw() {
    ctx.beginPath();
    ctx.fillStyle = this.color;
    if (this.shape === 'circle') {
      ctx.arc(this.x, this.y, 3, 0, Math.PI * 2);
    } else if (this.shape === 'square') {
      ctx.rect(this.x - 3, this.y - 3, 6, 6);
    } else if (this.shape === 'triangle') {
      ctx.moveTo(this.x, this.y - 3);
      ctx.lineTo(this.x - 3, this.y + 3);
      ctx.lineTo(this.x + 3, this.y + 3);
      ctx.closePath();
    } else if (this.shape === 'star') {
      for (let i = 0; i < 5; i++) {
        ctx.lineTo(
          this.x + 3 * Math.cos((18 + i * 72) * Math.PI / 180),
          this.y - 3 * Math.sin((18 + i * 72) * Math.PI / 180)
        );
        ctx.lineTo(
          this.x + 1.5 * Math.cos((54 + i * 72) * Math.PI / 180),
          this.y - 1.5 * Math.sin((54 + i * 72) * Math.PI / 180)
        );
      }
      ctx.closePath();
    }
    ctx.fill();
  }

  update() {
    if (window.innerWidth < 768) {
      const titleHeight = 50;
      const safeY = Math.min(Math.max(this.y, 10), 140);
      const safeX = Math.min(Math.max(this.x, 10), canvas.width - 10);
      this.x = safeX;
      this.y = safeY;
    } else {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
    }
  }
}

function animateDots() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  dots.forEach(dot => {
    dot.update();
    dot.draw();
  });
  requestAnimationFrame(animateDots);
}

function initDots(agent) {
  dots.length = 0;
  const numDots = 100;
  const isMobile = window.innerWidth < 768;
  if (agent === 'agentic') {
    for (let i = 0; i < numDots; i++) {
      const color = colors.agentic[i % 4];
      const shape = shapes[i % 4];
      const agentIdx = i % 4;
      let x, y;
      if (isMobile) {
        x = Math.random() * (canvas.width - 20) + 10;
        y = Math.random() * (140 - 10) + 10;
      } else {
        x = agentIdx % 2 ? canvas.width * 0.75 : canvas.width * 0.25;
        y = Math.random() * canvas.height;
      }
      dots.push(new Dot(x, y, color, shape, `agent${agentIdx + 1}`));
    }
  } else {
    const color = colors[agent];
    const shape = shapes[parseInt(agent.replace('agent', '')) - 1] || 'circle';
    for (let i = 0; i < numDots; i++) {
      let x, y;
      if (isMobile) {
        x = Math.random() * (canvas.width - 20) + 10;
        y = Math.random() * (140 - 10) + 10;
      } else {
        x = i % 2 ? canvas.width * 0.75 : canvas.width * 0.25;
        y = Math.random() * canvas.height;
      }
      dots.push(new Dot(x, y, color, shape, agent));
    }
  }
  animateDots();
}

window.setAgentsActive = function(active, agent) {
  if (active) {
    initDots(agent);
  } else {
    dots.length = 0;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
};
