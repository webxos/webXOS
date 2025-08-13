window.initNeurots = function() {
  const canvas = document.getElementById('neurots-visualization');
  if (!canvas) {
    console.error('Neurots canvas not found');
    return;
  }

  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const particles = [];
  const colors = ['#00ff00', '#00ffff', '#ff00ff', '#0000ff'];

  // Initialize particles for API metrics and Git commands
  function initParticles(data) {
    particles.length = 0;
    const count = Math.min(data.length || 50, 100);
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 4,
        vy: (Math.random() - 0.5) * 4,
        radius: Math.random() * 5 + 2,
        color: colors[Math.floor(Math.random() * colors.length)],
        type: data[i]?.endpoint || 'git'
      });
    }
  }

  // Animate particles with connections
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      // Bounce off walls
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      // Draw particle
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();

      // Draw connections
      particles.forEach(p2 => {
        const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(${parseInt(p.color.slice(1, 3), 16)}, ${parseInt(p.color.slice(3, 5), 16)}, ${parseInt(p.color.slice(5, 7), 16)}, ${1 - dist / 100})`;
          ctx.stroke();
        }
      });
    });
    requestAnimationFrame(animate);
  }

  // Fetch API metrics and initialize
  fetch('/v1/api/metrics', {
    headers: { 'Authorization': `Bearer ${localStorage.getItem('apiKey')}` }
  })
    .then(res => res.json())
    .then(data => {
      initParticles(data);
      animate();
    })
    .catch(err => console.error('Neurots metrics fetch failed:', err));
};
