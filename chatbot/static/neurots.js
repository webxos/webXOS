const canvas = document.createElement('canvas');
canvas.width = 800;
canvas.height = 400;
document.body.appendChild(canvas);
const ctx = canvas.getContext('2d');

function drawNeuralNetwork(nodes, connections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#007bff';
  ctx.fillStyle = '#007bff';

  connections.forEach(([from, to]) => {
    ctx.beginPath();
    ctx.moveTo(nodes[from].x, nodes[from].y);
    ctx.lineTo(nodes[to].x, nodes[to].y);
    ctx.stroke();
  });

  nodes.forEach(node => {
    ctx.beginPath();
    ctx.arc(node.x, node.y, 10, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function initNeuralViz() {
  try {
    const nodes = [
      { x: 100, y: 100 }, { x: 300, y: 100 }, { x: 500, y: 100 },
      { x: 200, y: 300 }, { x: 400, y: 300 }
    ];
    const connections = [[0, 3], [1, 3], [1, 4], [2, 4]];
    drawNeuralNetwork(nodes, connections);
  } catch (e) {
    console.error(`Neural viz error: ${e.message}`);
    fetch('/api/log-error', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: `Neural viz error: ${e.message}`, timestamp: new Date().toISOString() })
    });
  }
}

document.addEventListener('DOMContentLoaded', initNeuralViz);
