# Using Webxos

**Webxos** builds lightweight, single-page HTML apps for the decentralized web, using native JavaScript, WebGL, Three.js, and WebAssembly (WASM). Apps on [webxos.netlify.app](https://webxos.netlify.app) (e.g., eco-friendly tools, retro games, AI agents like FalseNode@webxos) are standalone `.html` files. This guide shows how to create and edit Webxos apps.

## Creating an App

1. **Start a New App**:
   - Create a `.html` file in `apps/` (e.g., `apps/MyApp.html`).
   - Include all HTML, CSS, JavaScript, and assets inline.

2. **Run It**:
   - Serve locally:
     ```bash
     npm run dev
     ```
   - Open `http://localhost:3000/apps/MyApp.html` in Chrome.
   - Or, open the `.html` file directly in Chrome.

## Example: Simple WebGL App

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Webxos WebGL Demo">
  <title>Webxos WebGL Demo</title>
  <style>
    body { margin: 0; background: #000; color: #00ff00; font-family: 'Courier New', monospace; }
    canvas { width: 100%; height: 100vh; }
    footer { font-size: 10px; text-align: center; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <footer>WebGL Demo v1.0.0 © 2025 WEBXOS</footer>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('canvas') });
    renderer.setSize(window.innerWidth, window.innerHeight);
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);
    camera.position.z = 5;
    function animate() {
      requestAnimationFrame(animate);
      cube.rotation.x += 0.01;
      cube.rotation.y += 0.01;
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
```

**Optional AI Prompt**:
```
Create a single-page HTML app with a rotating WebGL cube, neon-green style, and minimal code.
```

Save as `apps/WebGLDemo.html` and test in Chrome.

## Editing an App

- Open an existing `.html` file in `apps/` (e.g., `apps/FalseNode.html`).
- Modify HTML, CSS, or JavaScript inline.
- Add features like IPFS for P2P or WASM for performance.
- Test changes in Chrome (online/offline).

## Tips

- Keep apps lightweight: Avoid external frameworks.
- Use AI tools (Grok, ChatGPT, etc.) for quick code generation.
- Add meta tags and a footer for consistency.
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for submission guidelines.

## Resources

- Explore apps at [webxos.netlify.app](https://webxos.netlify.app).
- Check `apps/` in [webxos/webxos](https://github.com/webxos/webxos).
- Join [@webxos](https://x.com/webxos) or [GitHub Discussions](https://github.com/webxos/webxos/discussions).
