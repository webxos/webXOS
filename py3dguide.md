Guide to Building .py3d Files for Skulptor3D
This guide provides a comprehensive overview of creating .py3d files for the Skulptor3D environment, a lightweight 3D simulation platform with a custom parser. The .py3d format defines static scene elements (e.g., objects, initial animations, camera settings) that are loaded into Skulptor3D via the skulptor3d.html interface. This guide draws on insights from analyzing a Three.js-based simulation to enhance .py3d functionality, focusing on the drone swarm simulator (swarm.py3d) with features like LiDAR, forest rendering, and automation.
Overview of .py3d Format
The .py3d format is a simple, text-based scripting language with a limited set of commands for defining 3D scenes. It is parsed by skulptor3d.html and rendered using the js3d API. Key characteristics:

Commands: Limited to clear_scene, add_shape, animate, and set_camera.
Objects: Only supports cube shapes with uniform scaling (single size value).
Parameters:
position=(x,y,z): 3D coordinates as a tuple of floats.
size=float: Single scalar for uniform cube scaling (non-uniform tuples like (500,500,2) cause parsing errors).
color=(r,g,b): RGB values (0.0 to 1.0).
rotation=(rx,ry,rz): Euler angles in radians (for set_camera).


Constraints:
Strict syntax; deviations (e.g., invalid commands, non-scalar sizes) cause parsing errors (e.g., [ERROR] Unknown command).
Limited object count (typically <100) to avoid WebGL memory issues.


Purpose: Defines initial scene setup; dynamic behavior (e.g., automation, controls) is handled via JavaScript injection.

Steps to Build a .py3d File
Follow these steps to create a .py3d file, using the drone swarm simulator as an example.
1. Initialize the Scene

Command: clear_scene()
Purpose: Clears any existing objects to start with a blank canvas.
Example:clear_scene()


Notes:
Always include at the start to avoid conflicts with prior scenes.
Ensures consistent rendering when re-injecting the script.



2. Define Objects with add_shape

Syntax: add_shape(cube, name, position=(x,y,z), size=float, color=(r,g,b))
Purpose: Adds a cube object with a unique name, position, uniform size, and color.
Rules:
name: Unique string identifier (e.g., drone_0, tree_5).
position: Tuple of three floats for x, y, z coordinates.
size: Single float (e.g., 2.0). Non-uniform sizes (e.g., (500,500,2)) cause parsing errors.
color: Tuple of three floats (0.0 to 1.0) for RGB.


Example (from swarm.py3d):# Drones (neon green)
add_shape(cube, drone_0, position=(-5.0,2.5,-60.0), size=2.0, color=(0.0,1.0,0.0))
# Ground plane (green)
add_shape(cube, ground, position=(0.0,0.0,-50.0), size=200.0, color=(0.0,0.5,0.0))
# Tree (white)
add_shape(cube, tree_0, position=(-250.0,100.0,-25.0), size=0.3, color=(1.0,1.0,1.0))


Tips:
Use descriptive names (e.g., drone_0, tree_0) for JavaScript manipulation.
Limit object count (e.g., 50 trees) to avoid WebGL memory limits.
For large objects like ground planes, use a reasonable size (e.g., 200.0) to ensure visibility without overwhelming the renderer.
Scatter objects (e.g., trees) in a defined volume (e.g., 500x500x50) using random or fixed positions.



3. Add Initial Animations with animate

Syntax: animate(name, position=(x,y,z), duration=float)
Purpose: Defines initial animations for objects, moving them to a new position over a specified duration (in seconds).
Rules:
name: Matches an object defined by add_shape.
position: Target coordinates as a tuple.
duration: Float for animation length (e.g., 2.0 seconds).


Example:animate(drone_0, position=(-5.0,2.5,-40.0), duration=2.0)


Notes:
Animations run once at scene load.
Continuous animations (e.g., swarm movement) must be handled in JavaScript using js3d.animate.
Ensure animations don’t move objects out of the camera’s view.



4. Set the Camera with set_camera

Syntax: set_camera(position=(x,y,z), rotation=(rx,ry,rz))
Purpose: Sets the initial camera position and rotation (Euler angles in radians).
Rules:
position: Camera coordinates.
rotation: Euler angles (x, y, z) for orientation.


Example:set_camera(position=(-5.0,4.5,-30.0), rotation=(0.0,-0.2,0.0))


Tips:
Position the camera to view key objects (e.g., drones, ground, trees).
Use a slight downward tilt (e.g., ry=-0.2) to include the ground plane.
Update dynamically in JavaScript for first-person tracking.



5. Avoid Common Parsing Errors

Non-Uniform Scaling: Use a single size value (e.g., size=200.0), not tuples (e.g., size=(500,500,2)).
Invalid Commands: Stick to clear_scene, add_shape, animate, set_camera.
Syntax Errors: Ensure proper tuple formatting (e.g., (x,y,z)) and no extra spaces or characters.
Example Error Fix:
Invalid: add_shape(cube, ground, position=(0.0,0.0,-27.0), size=(500.0,500.0,2.0), color=(0.0,0.5,0.0))
Fixed: add_shape(cube, ground, position=(0.0,0.0,-50.0), size=200.0, color=(0.0,0.5,0.0))



6. Optimize for Performance

Limit Objects: Keep total objects (e.g., drones + trees + ground) under 100 to avoid WebGL memory issues.
Small Sizes: Use small sizes for numerous objects (e.g., size=0.3 for trees) to reduce rendering load.
Positioning: Place objects within the camera’s view (e.g., z=-50 to 0) to ensure visibility.

7. Enhance with JavaScript

Dynamic Behavior: Use JavaScript for continuous animations, controls, and interactions (e.g., LiDAR, automation).
Example Enhancements (inspired by Three.js simulation):
Console Commands: Add /add_node, /save_lidar for user interaction.
Object Pooling: Reuse LiDAR point clouds to reduce memory usage.
Frustum Culling: Hide objects outside the camera’s view for performance.
Prompts: Show temporary messages (e.g., “LiDAR scan completed”).
Status Overlay: Display real-time stats (e.g., drone count, LiDAR detections).
Autopilot: Navigate drones through waypoints (e.g., trees or user-added nodes).



Example: Building swarm.py3d
Here’s how to construct swarm.py3d for a drone swarm simulator with a forest, ground, and LiDAR.

Clear the Scene:
clear_scene()


Add Drones (8 neon green cubes):
add_shape(cube, drone_0, position=(-5.0,2.5,-60.0), size=2.0, color=(0.0,1.0,0.0))
add_shape(cube, drone_1, position=(5.0,2.5,-60.0), size=2.0, color=(0.0,1.0,0.0))
# ... (add drone_2 to drone_7 similarly)


Add Ground Plane (large green cube):
add_shape(cube, ground, position=(0.0,0.0,-50.0), size=200.0, color=(0.0,0.5,0.0))


Add Forest (50 white cube trees):
add_shape(cube, tree_0, position=(-250.0,100.0,-25.0), size=0.3, color=(1.0,1.0,1.0))
add_shape(cube, tree_1, position=(200.0,-150.0,0.0), size=0.3, color=(1.0,1.0,1.0))
# ... (add tree_2 to tree_49 with random or fixed positions in 500x500x50 volume)


Generate random positions:import random
for i in range(2, 50):
    x = (random.random() - 0.5) * 500
    y = (random.random() - 0.5) * 500
    z = (random.random() - 0.5) * 50
    print(f"add_shape(cube, tree_{i}, position=({x:.1f},{y:.1f},{z:.1f}), size=0.3, color=(1.0,1.0,1.0))")




Add Initial Animations:
animate(drone_0, position=(-5.0,2.5,-40.0), duration=2.0)
animate(drone_1, position=(5.0,2.5,-40.0), duration=2.0)
# ... (add for drone_2 to drone_7)


Set Camera:
set_camera(position=(-5.0,4.5,-30.0), rotation=(0.0,-0.2,0.0))



Testing and Troubleshooting

Inject the Script:

Open skulptor3d.html in Chrome (10:15 PM EDT, July 23, 2025).
Click "Inject," select "Skulptor3D (.py3d)," paste the .py3d content, and click "Inject."
Check console for parsing errors (e.g., [ERROR] Unknown command).


Verify Scene:

Enter /show3d to open the 3D canvas.
Confirm objects (drones, ground, trees) are visible.
Check console for "[INFO] Scene objects: 8 drones, 50 trees, 1 ground".


Common Issues:

Parsing Errors: Verify size is a scalar, commands are valid, and syntax is correct.
Objects Not Visible:
Adjust camera position: js3d.set_camera([-5, 4.5, -30], [0, -0.2, 0]).
Test with a single cube: js3d.add_cube('test_cube', [0, 0, -40], 1, [1, 1, 1]).
Reduce object count if WebGL errors occur.


Animations: Ensure JavaScript uses js3d.animate for continuous motion to avoid [CHECK 5] Animation: No animations.



JavaScript Enhancements
To fully realize the simulation, inject a JavaScript script after loading .py3d. Key features to implement:

Controls: Arrow keys for lead drone movement, with camera tracking.
Automation: PSO-inspired swarm formation and waypoint navigation (e.g., /add_node).
LiDAR: Simulate point clouds with object pooling.
UI: Add buttons, status overlay, prompts, and console commands.
Performance: Use frustum culling and limit LiDAR scans (e.g., every 0.5s).

Example JavaScript (Snippet)
// Add node command
function addNode() {
    const nodeId = `node_${nodes.length}`;
    const nodePos = [leadDronePosition[0], leadDronePosition[1], leadDronePosition[2] - 50];
    js3d.add_cube(nodeId, nodePos, 1, [1, 0, 1]);
    nodes.push({ id: nodeId, position: nodePos });
    showPrompt('Navigation node added.');
}

Best Practices

Modularity: Keep .py3d for static setup; use JavaScript for dynamics.
Performance: Limit objects, reuse point clouds, and cull off-screen objects.
User Feedback: Use prompts and status overlays for clarity.
Testing: Inject incrementally, checking console logs for errors.
Extensibility: Add commands like /save_lidar for training data export.
