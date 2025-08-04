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

class Dot {
  constructor(x, y, color, pattern) {
    this.x = x;
    this.y = y;
    this.color = color;
    this.pattern = pattern;
    this.vx = (Math.random() - 0.5) * 2;
    this.vy = (Math.random() - 0.5) * 2;
  }

  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
  }

  update() {
    if (window.innerWidth < 768) {
      // Mobile: Constrain to 150px-high area around logo
      const titleHeight = 50; // Approximate height of h1.title
      const safeY = Math.min(Math.max(this.y, 10), 140); // Stay within 10-140px from top
      const safeX = Math.min(Math.max(this.x, 10), canvas.width - 10); // Avoid edges
      this.x = safeX;
      this.y = safeY;
    } else {
      // Desktop: Existing behavior
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
  const titleHeight = 50; // Approximate height of h1.title
  if (agent === 'agentic') {
    for (let i = 0; i < numDots; i++) {
      const color = colors.agentic[i % 4];
      const pattern = i % 4 === 0 ? 'helix' : i % 4 === 1 ? 'grid' : i % 4 === 2 ? 'torus' : 'swarm';
      let x, y;
      if (isMobile) {
        // Randomly place around logo, avoiding overlap
        x = Math.random() * (canvas.width - 20) + 10;
        y = Math.random() * (140 - 10) + 10; // 10-140px from top
      } else {
        x = i % 2 ? canvas.width * 0.75 : canvas.width * 0.25;
        y = Math.random() * canvas.height;
      }
      dots.push(new Dot(x, y, color, pattern));
    }
  } else {
    const color = colors[agent];
    for (let i = 0; i < numDots; i++) {
      let x, y;
      if (isMobile) {
        x = Math.random() * (canvas.width - 20) + 10;
        y = Math.random() * (140 - 10) + 10;
      } else {
        x = i % 2 ? canvas.width * 0.75 : canvas.width * 0.25;
        y = Math.random() * canvas.height;
      }
      dots.push(new Dot(x, y, color, agent));
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
