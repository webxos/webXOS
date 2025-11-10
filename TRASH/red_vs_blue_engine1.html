<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>GALAXYCRAFT: RED VS BLUE | Team Drone Combat</title>
  <meta name="description" content="Experience intense team-based FPS combat in the GalaxyCraft universe with Red vs Blue drone warfare.">
  <meta name="keywords" content="GalaxyCraft, FPS, team combat, base defense, WebGL, drone swarm, Red vs Blue">
  <meta name="author" content="WebXOS">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Orbitron', 'Courier New', monospace;
    }
    body {
      background: #0a0a2a;
      color: #f55;
      height: 100vh;
      overflow: hidden;
      touch-action: none;
      cursor: none;
      background-image: 
        radial-gradient(circle at 20% 30%, rgba(51, 102, 255, 0.2) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(255, 51, 51, 0.2) 0%, transparent 40%),
        linear-gradient(to bottom, #0a0a2a, #000020);
    }
    #gameCanvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 0;
    }
    
    /* Crosshair */
    .crosshair {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 30px;
      height: 30px;
      z-index: 10;
      pointer-events: none;
    }
    .crosshair::before, .crosshair::after {
      content: '';
      position: absolute;
      background: #f55;
    }
    .crosshair::before {
      width: 20px;
      height: 2px;
      left: 5px;
      top: 14px;
    }
    .crosshair::after {
      width: 2px;
      height: 20px;
      left: 14px;
      top: 5px;
    }
    .crosshair.dot::before {
      width: 4px;
      height: 4px;
      left: 13px;
      top: 13px;
      border-radius: 50%;
    }
    .crosshair.dot::after {
      display: none;
    }
    
    /* Enhanced scope crosshair */
    .scope-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle, transparent 40%, rgba(0, 0, 0, 0.8) 40.5%);
      z-index: 9;
      pointer-events: none;
      display: none;
    }
    .scope-crosshair {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 100px;
      height: 100px;
      z-index: 10;
      pointer-events: none;
      display: none;
    }
    .scope-crosshair::before, .scope-crosshair::after {
      content: '';
      position: absolute;
      background: #f00;
    }
    .scope-crosshair::before {
      width: 80px;
      height: 2px;
      left: 10px;
      top: 49px;
    }
    .scope-crosshair::after {
      width: 2px;
      height: 80px;
      left: 49px;
      top: 10px;
    }
    .scope-dot {
      position: absolute;
      width: 6px;
      height: 6px;
      background: #f00;
      border-radius: 50%;
      top: 47px;
      left: 47px;
      box-shadow: 0 0 10px rgba(255, 0, 0, 0.8);
    }
    
    /* Weapon HUD */
    .weapon-hud {
      position: fixed;
      bottom: 120px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      flex-direction: column;
      align-items: center;
      z-index: 20;
    }
    .weapon-icon {
      font-size: 40px;
      margin-bottom: 10px;
      text-shadow: 0 0 10px rgba(255, 85, 85, 0.8);
    }
    .ammo-display {
      font-size: 24px;
      text-shadow: 0 0 5px rgba(255, 85, 85, 0.8);
    }
    
    /* Main UI */
    .ui-container {
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 15px;
      z-index: 20;
      padding: 15px;
      background: rgba(30, 10, 20, 0.7);
      border-radius: 20px;
      border: 2px solid #f55;
      backdrop-filter: blur(5px);
      transition: all 0.5s ease;
    }
    .ui-btn {
      width: 70px;
      height: 70px;
      border-radius: 50%;
      background: linear-gradient(145deg, #502020, #301010);
      border: 2px solid #f55;
      color: #f55;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      box-shadow: 0 0 15px rgba(255, 85, 85, 0.5);
      transition: all 0.2s ease;
    }
    .ui-btn:hover {
      background: linear-gradient(145deg, #603030, #402020);
      box-shadow: 0 0 20px rgba(255, 85, 85, 0.8);
    }
    .ui-btn.active {
      background: linear-gradient(145deg, #ff3300, #cc2200);
      border-color: #ff3300;
      box-shadow: 0 0 20px rgba(255, 51, 0, 0.8);
      color: #fff;
    }
    .ui-btn .key {
      font-size: 20px;
      font-weight: bold;
      margin-bottom: 4px;
    }
    
    /* Combat Stats */
    .combat-stats {
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(0, 0, 0, 0.7);
      padding: 15px;
      border: 2px solid #f55;
      border-radius: 10px;
      font-size: 16px;
      color: #f55;
      z-index: 15;
    }
    .health-bar, .shield-bar, .energy-bar {
      width: 200px;
      height: 20px;
      background: #333;
      border-radius: 10px;
      margin: 10px 0;
      overflow: hidden;
    }
    .health-fill {
      height: 100%;
      width: 100%;
      background: linear-gradient(90deg, #f00, #ff3300);
      border-radius: 10px;
      transition: width 0.3s ease;
    }
    .shield-fill {
      height: 100%;
      width: 100%;
      background: linear-gradient(90deg, #55f, #33f);
      border-radius: 10px;
      transition: width 0.3s ease;
    }
    .energy-fill {
      height: 100%;
      width: 100%;
      background: linear-gradient(90deg, #0f0, #0c0);
      border-radius: 10px;
      transition: width 0.3s ease;
    }
    .warning-alert {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 36px;
      color: #f00;
      text-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
      text-align: center;
      opacity: 0;
      transition: opacity 0.3s ease;
      z-index: 25;
      pointer-events: none;
    }
    .warning-alert.active {
      animation: alert-flash 0.5s infinite alternate;
    }
    @keyframes alert-flash {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    /* Currency Display */
    .currency-display {
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.7);
      padding: 15px;
      border: 2px solid #f55;
      border-radius: 10px;
      font-size: 16px;
      color: #f55;
      z-index: 15;
      display: flex;
      align-items: center;
    }
    .currency-icon {
      color: #f55;
      font-size: 20px;
      margin-right: 10px;
    }
    
    /* Team Status */
    .team-status {
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 50px;
      background: rgba(0, 0, 0, 0.7);
      padding: 10px 20px;
      border-radius: 10px;
      z-index: 15;
    }
    .team-blue, .team-red {
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .team-blue {
      color: #55f;
      border: 2px solid #55f;
      padding: 5px 15px;
      border-radius: 5px;
    }
    .team-red {
      color: #f55;
      border: 2px solid #f55;
      padding: 5px 15px;
      border-radius: 5px;
    }
    .team-health {
      font-size: 18px;
      font-weight: bold;
      margin-top: 5px;
    }
    
    /* Main Menu */
    .main-menu {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(10, 10, 30, 0.95);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 100;
      color: #f55;
      cursor: default;
      background-image: 
        radial-gradient(circle at 20% 30%, rgba(51, 102, 255, 0.3) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(255, 51, 51, 0.3) 0%, transparent 40%),
        linear-gradient(to bottom, #0a0a2a, #000020);
    }
    .game-title {
      font-size: 60px;
      margin-bottom: 20px;
      text-shadow: 0 0 20px rgba(255, 85, 85, 0.8);
      letter-spacing: 4px;
      text-align: center;
      background: linear-gradient(to right, #55f, #f55);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .game-subtitle {
      font-size: 24px;
      margin-bottom: 40px;
      text-shadow: 0 0 10px rgba(255, 85, 85, 0.8);
      letter-spacing: 2px;
    }
    .menu-options {
      display: flex;
      flex-direction: column;
      gap: 20px;
      width: 300px;
    }
    .menu-btn {
      padding: 15px 30px;
      background: linear-gradient(145deg, #302050, #201040);
      border: 2px solid #f55;
      color: #f55;
      font-size: 20px;
      text-align: center;
      cursor: pointer;
      border-radius: 10px;
      transition: all 0.3s ease;
    }
    .menu-btn:hover {
      background: linear-gradient(145deg, #403060, #302050);
      box-shadow: 0 0 20px rgba(255, 85, 85, 0.8);
      transform: translateY(-3px);
    }
    .settings-panel {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(30, 20, 40, 0.9);
      padding: 30px;
      border: 2px solid #f55;
      border-radius: 15px;
      width: 400px;
      display: none;
      z-index: 110;
      cursor: default;
    }
    .settings-panel h2 {
      margin-bottom: 20px;
      text-align: center;
    }
    .settings-row {
      display: flex;
      justify-content: space-between;
      margin: 15px 0;
    }
    .settings-row label {
      font-size: 18px;
    }
    .settings-row input {
      background: #402020;
      border: 1px solid #f55;
      color: #f55;
      padding: 5px 10px;
      border-radius: 5px;
      width: 80px;
    }
    .close-settings {
      position: absolute;
      top: 10px;
      right: 10px;
      background: none;
      border: none;
      color: #f55;
      font-size: 20px;
      cursor: pointer;
    }
    
    /* Stats Panel */
    .stats-panel {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(30, 20, 40, 0.9);
      padding: 30px;
      border: 2px solid #f55;
      border-radius: 15px;
      width: 400px;
      display: none;
      z-index: 110;
      cursor: default;
    }
    .stats-panel h2 {
      margin-bottom: 20px;
      text-align: center;
    }
    .stats-row {
      display: flex;
      justify-content: space-between;
      margin: 10px 0;
      font-size: 18px;
    }
    
    /* Console Messages */
    .console-message {
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.8);
      padding: 10px 20px;
      border: 2px solid #f55;
      border-radius: 10px;
      color: #f55;
      font-size: 18px;
      opacity: 0;
      transition: opacity 0.5s ease;
      z-index: 25;
    }
    .console-message.active {
      opacity: 1;
    }
    
    /* Hit Marker */
    .hit-marker {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 40px;
      height: 40px;
      border: 2px solid #f55;
      border-radius: 50%;
      z-index: 10;
      pointer-events: none;
      opacity: 0;
    }
    .hit-marker.active {
      animation: hitMarker 0.3s ease-out;
    }
    @keyframes hitMarker {
      0% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
      100% { opacity: 0; transform: translate(-50%, -50%) scale(1.5); }
    }
    
    /* Damage Effect */
    .damage-effect {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(255, 0, 0, 0.2);
      z-index: 5;
      pointer-events: none;
      opacity: 0;
    }
    .damage-effect.active {
      animation: damageFlash 0.3s ease-out;
    }
    @keyframes damageFlash {
      0% { opacity: 0.5; }
      100% { opacity: 0; }
    }
    
    /* Team Selection */
    .team-selection {
      display: flex;
      gap: 20px;
      margin-bottom: 20px;
    }
    .team-btn {
      padding: 10px 20px;
      border: 2px solid;
      border-radius: 5px;
      cursor: pointer;
      font-size: 18px;
      transition: all 0.3s ease;
    }
    .team-blue-btn {
      color: #55f;
      border-color: #55f;
      background: rgba(0, 0, 0, 0.5);
    }
    .team-blue-btn.active, .team-blue-btn:hover {
      background: rgba(0, 0, 255, 0.2);
    }
    .team-red-btn {
      color: #f55;
      border-color: #f55;
      background: rgba(0, 0, 0, 0.5);
    }
    .team-red-btn.active, .team-red-btn:hover {
      background: rgba(255, 0, 0, 0.2);
    }
    
    /* Jetpack Fuel */
    .jetpack-fuel {
      position: absolute;
      bottom: 180px;
      right: 20px;
      background: rgba(0, 0, 0, 0.7);
      padding: 10px;
      border: 2px solid #5af;
      border-radius: 10px;
      font-size: 14px;
      color: #5af;
      z-index: 15;
      width: 150px;
    }
    .fuel-bar {
      width: 100%;
      height: 15px;
      background: #333;
      border-radius: 7px;
      margin-top: 5px;
      overflow: hidden;
    }
    .fuel-fill {
      height: 100%;
      width: 100%;
      background: linear-gradient(90deg, #5af, #07f);
      border-radius: 7px;
      transition: width 0.3s ease;
    }
    
    /* Jetpack Effect */
    .jetpack-effect {
      position: absolute;
      width: 20px;
      height: 40px;
      background: linear-gradient(to top, rgba(255, 170, 0, 0.8), transparent);
      border-radius: 50% 50% 0 0;
      z-index: 5;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    
    /* Scan Effect */
    .scan-effect {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 8;
      pointer-events: none;
      opacity: 0;
      background: radial-gradient(circle at center, transparent 30%, rgba(255, 0, 0, 0.2) 100%);
    }
    .scan-effect.active {
      animation: scanPulse 2s ease-out;
    }
    @keyframes scanPulse {
      0% { opacity: 0.5; transform: scale(0.5); }
      100% { opacity: 0; transform: scale(1.5); }
    }
    
    /* Target Indicators */
    .target-indicator {
      position: fixed;
      width: 20px;
      height: 20px;
      border: 2px solid #f00;
      border-radius: 50%;
      z-index: 7;
      pointer-events: none;
      box-shadow: 0 0 10px rgba(255, 0, 0, 0.8);
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    
    /* Shield Effect */
    .shield-effect {
      position: absolute;
      width: 100px;
      height: 100px;
      border-radius: 50%;
      border: 3px solid rgba(85, 85, 255, 0.7);
      box-shadow: 0 0 20px rgba(85, 85, 255, 0.5);
      z-index: 6;
      pointer-events: none;
      opacity: 0;
      transition: all 0.3s ease;
    }
    
    /* Heal Effect */
    .heal-effect {
      position: absolute;
      width: 50px;
      height: 50px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(0, 255, 0, 0.5) 0%, transparent 70%);
      z-index: 6;
      pointer-events: none;
      opacity: 0;
      transition: all 0.3s ease;
    }
    
    /* Class Selection */
    .class-selection {
      display: flex;
      gap: 15px;
      margin-bottom: 20px;
    }
    .class-btn {
      padding: 10px 15px;
      border: 2px solid #f55;
      border-radius: 5px;
      cursor: pointer;
      font-size: 16px;
      transition: all 0.3s ease;
      background: rgba(0, 0, 0, 0.5);
      color: #f55;
    }
    .class-btn.active, .class-btn:hover {
      background: rgba(255, 0, 0, 0.2);
      box-shadow: 0 0 10px rgba(255, 85, 85, 0.8);
    }
    
    /* Stars Background */
    .stars {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: -1;
      pointer-events: none;
    }
    .star {
      position: absolute;
      background: #fff;
      border-radius: 50%;
      animation: twinkle linear infinite;
    }
    @keyframes twinkle {
      0%, 100% { opacity: 0.2; }
      50% { opacity: 1; }
    }
    
    /* Copyright Notice */
    .copyright {
      position: absolute;
      bottom: 10px;
      left: 50%;
      transform: translateX(-50%);
      color: #888;
      font-size: 12px;
      text-align: center;
      z-index: 101;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
      .ui-container {
        bottom: 10px;
        padding: 10px;
        gap: 10px;
      }
      .ui-btn {
        width: 60px;
        height: 60px;
        font-size: 10px;
      }
      .ui-btn .key {
        font-size: 16px;
      }
      .game-title {
        font-size: 40px;
      }
      .game-subtitle {
        font-size: 18px;
      }
      .menu-btn {
        padding: 12px 24px;
        font-size: 18px;
      }
      .combat-stats {
        bottom: 100px;
        left: 10px;
        font-size: 14px;
        padding: 10px;
      }
      .health-bar, .shield-bar, .energy-bar {
        width: 150px;
        height: 15px;
      }
      .team-status {
        flex-direction: column;
        gap: 10px;
        top: 80px;
      }
      .weapon-hud {
        bottom: 100px;
      }
      .weapon-icon {
        font-size: 30px;
      }
      .ammo-display {
        font-size: 18px;
      }
      .jetpack-fuel {
        bottom: 160px;
        right: 10px;
        width: 120px;
      }
      .class-selection {
        flex-direction: column;
        gap: 10px;
      }
    }
  </style>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
</head>
<body>
  <div class="stars" id="stars"></div>
  
  <div id="gameContainer">
    <canvas id="gameCanvas"></canvas>
    
    <!-- Crosshair -->
    <div class="crosshair" id="crosshair"></div>
    
    <!-- Scope Elements -->
    <div class="scope-overlay" id="scopeOverlay"></div>
    <div class="scope-crosshair" id="scopeCrosshair">
      <div class="scope-dot"></div>
    </div>
    
    <!-- Weapon HUD -->
    <div class="weapon-hud" id="weaponHud">
      <div class="weapon-icon" id="weaponIcon">🔫</div>
      <div class="ammo-display" id="ammoDisplay">30/90</div>
    </div>
    
    <!-- Hit Marker -->
    <div class="hit-marker" id="hitMarker"></div>
    
    <!-- Damage Effect -->
    <div class="damage-effect" id="damageEffect"></div>
    
    <!-- Scan Effect -->
    <div class="scan-effect" id="scanEffect"></div>
    
    <!-- Shield Effect -->
    <div class="shield-effect" id="shieldEffect"></div>
    
    <!-- Heal Effect -->
    <div class="heal-effect" id="healEffect"></div>
    
    <!-- Jetpack Fuel -->
    <div class="jetpack-fuel" id="jetpackFuelDisplay">
      <div>JETPACK FUEL</div>
      <div class="fuel-bar"><div class="fuel-fill" id="fuelFill" style="width: 100%"></div></div>
    </div>
    
    <!-- Team Status -->
    <div class="team-status" id="teamStatus">
      <div class="team-blue">
        <div>BLUE BASE</div>
        <div class="team-health" id="blueBaseHealth">100%</div>
      </div>
      <div class="team-red">
        <div>RED BASE</div>
        <div class="team-health" id="redBaseHealth">100%</div>
      </div>
    </div>
    
    <!-- Combat Stats -->
    <div class="combat-stats">
      <div>WAVE: <span id="waveCount">1</span></div>
      <div>ENEMIES: <span id="enemyCount">0</span></div>
      <div>SCORE: <span id="combatScore">0</span></div>
      <div>HEALTH:</div>
      <div class="health-bar"><div class="health-fill" id="healthFill" style="width: 100%"></div></div>
      <div>SHIELD:</div>
      <div class="shield-bar"><div class="shield-fill" id="shieldFill" style="width: 100%"></div></div>
      <div>ENERGY:</div>
      <div class="energy-bar"><div class="energy-fill" id="energyFill" style="width: 100%"></div></div>
    </div>
    
    <!-- Main UI Controls -->
    <div class="ui-container">
      <div class="ui-btn" id="moveBtn">
        <div class="key">W</div>
        <div>MOVE</div>
      </div>
      <div class="ui-btn" id="jumpBtn">
        <div class="key">SHFT</div>
        <div id="jumpBtnLabel">JETPACK</div>
      </div>
      <div class="ui-btn" id="reloadBtn">
        <div class="key">R</div>
        <div>RELOAD</div>
      </div>
      <div class="ui-btn" id="fireBtn">
        <div class="key">SPC</div>
        <div id="fireBtnLabel">FIRE</div>
      </div>
      <div class="ui-btn" id="sprintBtn">
        <div class="key">CTRL</div>
        <div id="sprintBtnLabel">CROUCH</div>
      </div>
      <div class="ui-btn" id="menuBtn">
        <div class="key">ESC</div>
        <div>MENU</div>
      </div>
    </div>
    
    <!-- Console Messages -->
    <div class="console-message" id="consoleMessage"></div>
    
    <!-- Main Menu -->
    <div class="main-menu" id="mainMenu">
      <h1 class="game-title">GALAXYCRAFT</h1>
      <div class="game-subtitle">RED VS BLUE</div>
      
      <div class="team-selection">
        <div class="team-btn team-blue-btn active" id="blueTeamBtn">BLUE TEAM</div>
        <div class="team-btn team-red-btn" id="redTeamBtn">RED TEAM</div>
      </div>
      
      <div class="class-selection">
        <div class="class-btn active" id="scoutClassBtn">SCOUT</div>
        <div class="class-btn" id="assaultClassBtn">ASSAULT</div>
        <div class="class-btn" id="medicClassBtn">MEDIC</div>
      </div>
      
      <div class="menu-options">
        <div class="menu-btn" id="startBtn">JOIN BATTLE</div>
        <div class="menu-btn" id="statsBtn">STATISTICS</div>
        <div class="menu-btn" id="settingsBtn">SETTINGS</div>
        <div class="menu-btn" id="quitBtn">QUIT</div>
      </div>
      
      <div class="copyright">WebXOS 2025 © All Rights Reserved</div>
      
      <div class="settings-panel" id="settingsPanel">
        <button class="close-settings" id="closeSettings">X</button>
        <h2>GAME SETTINGS</h2>
        <div class="settings-row">
          <label>Mouse Sensitivity:</label>
          <input type="number" id="mouseSensitivity" value="0.003" step="0.001" min="0.001" max="0.01">
        </div>
        <div class="settings-row">
          <label>Invert Y-Axis:</label>
          <input type="checkbox" id="invertYAxis">
        </div>
        <div class="settings-row">
          <label>Invert S/W Keys:</label>
          <input type="checkbox" id="invertSWKeys">
        </div>
        <div class="settings-row">
          <label>FOV:</label>
          <input type="range" id="fovSetting" min="60" max="110" value="90">
        </div>
        <div class="settings-row">
          <label>Sound Volume:</label>
          <input type="range" id="soundVolume" min="0" max="100" value="80">
        </div>
      </div>
      
      <div class="stats-panel" id="statsPanel">
        <button class="close-settings" id="closeStats">X</button>
        <h2>PLAYER STATISTICS</h2>
        <div class="stats-row">
          <span>Kills:</span>
          <span id="statKills">0</span>
        </div>
        <div class="stats-row">
          <span>Deaths:</span>
          <span id="statDeaths">0</span>
        </div>
        <div class="stats-row">
          <span>Score:</span>
          <span id="statScore">0</span>
        </div>
        <div class="stats-row">
          <span>Damage Dealt:</span>
          <span id="statDamage">0</span>
        </div>
        <div class="stats-row">
          <span>Accuracy:</span>
          <span id="statAccuracy">0%</span>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
  <script>
    // Create stars background
    function createStars() {
      const stars = document.getElementById('stars');
      const count = 200;
      
      for (let i = 0; i < count; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        
        const size = Math.random() * 3;
        star.style.width = `${size}px`;
        star.style.height = `${size}px`;
        
        star.style.left = `${Math.random() * 100}%`;
        star.style.top = `${Math.random() * 100}%`;
        
        star.style.animationDuration = `${2 + Math.random() * 5}s`;
        star.style.animationDelay = `${Math.random() * 5}s`;
        
        stars.appendChild(star);
      }
    }
    
    // Game state and configuration
    const GameState = {
      MENU: 0,
      PLAYING: 1,
      PAUSED: 2
    };
    
    const PlayerClass = {
      SCOUT: 0,
      ASSAULT: 1,
      MEDIC: 2
    };
    
    let currentState = GameState.MENU;
    let mouseSensitivity = 0.003;
    let invertYAxis = false;
    let invertSWKeys = false;
    let fov = 90;
    let playerTeam = 'blue'; // Default team
    let playerClass = PlayerClass.SCOUT; // Default class
    let isPointerLocked = false;
    
    // Initialize Three.js
    let scene, camera, renderer;
    let player, drones = [], projectiles = [], terrain = [], bases = [];
    let clock = new THREE.Clock();
    
    // Player controls state
    const controls = {
      forward: false,
      backward: false,
      left: false,
      right: false,
      jump: false,
      sprint: false,
      fire: false,
      reload: false,
      isGrounded: false,
      jetpack: false,
      crouch: false,
      scan: false,
      scope: false,
      leanLeft: false,
      leanRight: false,
      shield: false,
      heal: false
    };
    
    // Class-specific configurations
    const classConfig = {
      [PlayerClass.SCOUT]: {
        name: "Scout",
        health: 100,
        shield: 50,
        energy: 100,
        speed: 6,
        sprintSpeed: 9,
        jumpStrength: 8,
        jetpackStrength: 250,
        jetpackFuel: 100,
        jetpackFuelMax: 100,
        jetpackFuelConsumption: 15,
        jetpackFuelRegen: 12,
        energyRegen: 5,
        hasJetpack: true,
        hasShield: false,
        hasHeal: false,
        primaryWeapon: {
          name: "Pulse Rifle",
          damage: 100,
          fireRate: 0.1,
          ammo: 30,
          maxAmmo: 90,
          reloadTime: 1.5,
          range: 100,
          icon: "🔫"
        },
        specialAbility: {
          name: "Scan",
          energyCost: 30,
          cooldown: 10,
          duration: 5
        }
      },
      [PlayerClass.ASSAULT]: {
        name: "Assault",
        health: 150,
        shield: 100,
        energy: 80,
        speed: 5,
        sprintSpeed: 7,
        jumpStrength: 6,
        jetpackStrength: 0,
        jetpackFuel: 0,
        jetpackFuelMax: 0,
        jetpackFuelConsumption: 0,
        jetpackFuelRegen: 0,
        energyRegen: 4,
        hasJetpack: false,
        hasShield: true,
        hasHeal: false,
        primaryWeapon: {
          name: "Shotgun",
          damage: 200,
          fireRate: 0.8,
          ammo: 6,
          maxAmmo: 36,
          reloadTime: 2.5,
          range: 30,
          icon: "🔫"
        },
        specialAbility: {
          name: "Shield",
          energyCost: 20,
          cooldown: 5,
          duration: 3
        }
      },
      [PlayerClass.MEDIC]: {
        name: "Medic",
        health: 120,
        shield: 80,
        energy: 120,
        speed: 5.5,
        sprintSpeed: 7.5,
        jumpStrength: 7,
        jetpackStrength: 0,
        jetpackFuel: 0,
        jetpackFuelMax: 0,
        jetpackFuelConsumption: 0,
        jetpackFuelRegen: 0,
        energyRegen: 6,
        hasJetpack: false,
        hasShield: true,
        hasHeal: true,
        primaryWeapon: {
          name: "Bio Rifle",
          damage: 80,
          fireRate: 0.2,
          ammo: 20,
          maxAmmo: 100,
          reloadTime: 1.8,
          range: 80,
          icon: "🔫"
        },
        specialAbility: {
          name: "Heal",
          energyCost: 25,
          cooldown: 3,
          duration: 2
        }
      }
    };
    
    // Initialize the game
    function init() {
      createStars();
      
      // Set up scene with optimized settings
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0a2a);
      
      // Add space background effect
      scene.background = new THREE.Color(0x0a0a2a);
      
      // Set up camera - First person view
      camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, 0.1, 500);
      camera.rotation.order = 'YXZ'; // Important for FPS controls
      
      // Set up renderer with performance optimizations
      renderer = new THREE.WebGLRenderer({ 
        canvas: document.getElementById('gameCanvas'),
        antialias: false,
        powerPreference: "high-performance"
      });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
      renderer.shadowMap.enabled = false;
      
      // Add lighting
      const ambientLight = new THREE.AmbientLight(0x554433);
      scene.add(ambientLight);
      
      const directionalLight = new THREE.DirectionalLight(0xffddcc, 0.7);
      directionalLight.position.set(5, 10, 5);
      scene.add(directionalLight);
      
      // Create mars-like terrain with optimized geometry
      createMarsTerrain();
      
      // Create bases
      createBases();
      
      // Create player
      createPlayer();
      
      // Create drones for both teams
      spawnDrones();
      
      // Create jetpack effect
      createJetpackEffect();
      
      // Set up event listeners
      setupEventListeners();
      
      // Start animation loop
      animate();
      
      // Show console message
      showConsoleMessage("GalaxyCraft: Red vs Blue initialized. Ready for combat!");
    }
    
    // Create mars-like terrain with optimized geometry
    function createMarsTerrain() {
      // Create ground plane with fewer segments for performance
      const groundGeometry = new THREE.PlaneGeometry(400, 200, 20, 10); // Rectangular map
      
      const groundMaterial = new THREE.MeshStandardMaterial({
        color: 0xaa4422,
        roughness: 0.8,
        metalness: 0.2
      });
      
      const ground = new THREE.Mesh(groundGeometry, groundMaterial);
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = -5;
      scene.add(ground);
      terrain.push(ground);
      
      // Add mountains on the edges for cover
      for (let i = 0; i < 20; i++) {
        const mountainSize = 5 + Math.random() * 15;
        const mountainGeometry = new THREE.ConeGeometry(mountainSize, 10 + Math.random() * 20, 8);
        const mountainMaterial = new THREE.MeshStandardMaterial({
          color: 0x773322,
          roughness: 0.9,
          metalness: 0.1
        });
        
        const mountain = new THREE.Mesh(mountainGeometry, mountainMaterial);
        
        // Place mountains along the edges of the map
        const side = Math.random() > 0.5 ? 1 : -1;
        const edge = Math.random() > 0.5 ? 'width' : 'height';
        
        if (edge === 'width') {
          mountain.position.set(
            side * 180,
            -5 + mountainSize/2,
            (Math.random() - 0.5) * 180
          );
        } else {
          mountain.position.set(
            (Math.random() - 0.5) * 180,
            -5 + mountainSize/2,
            side * 80
          );
        }
        
        scene.add(mountain);
        terrain.push(mountain);
      }
      
      // Add some structures for cover with reduced count
      for (let i = 0; i < 15; i++) {
        const structureWidth = 5 + Math.random() * 8;
        const structureHeight = 3 + Math.random() * 5;
        const structureDepth = 5 + Math.random() * 8;
        
        const structureGeometry = new THREE.BoxGeometry(structureWidth, structureHeight, structureDepth);
        const structureMaterial = new THREE.MeshStandardMaterial({
          color: 0x884422,
          roughness: 0.9,
          metalness: 0.1
        });
        
        const structure = new THREE.Mesh(structureGeometry, structureMaterial);
        structure.position.set(
          (Math.random() - 0.5) * 300,
          -5 + structureHeight/2,
          (Math.random() - 0.5) * 100
        );
        scene.add(structure);
        terrain.push(structure);
      }
    }
    
    // Create bases
    function createBases() {
      // Player base (blue)
      const blueBaseGeometry = new THREE.CylinderGeometry(12, 16, 8, 8);
      const blueBaseMaterial = new THREE.MeshStandardMaterial({
        color: 0x3355ff,
        emissive: 0x1133aa,
        metalness: 0.7,
        roughness: 0.3
      });
      
      const blueBase = new THREE.Mesh(blueBaseGeometry, blueBaseMaterial);
      blueBase.position.set(-150, 0, 0);
      blueBase.isPlayerBase = true;
      blueBase.team = 'blue';
      blueBase.health = 150000;
      scene.add(blueBase);
      bases.push(blueBase);
      
      // Enemy base (red)
      const redBaseGeometry = new THREE.CylinderGeometry(12, 16, 8, 8);
      const redBaseMaterial = new THREE.MeshStandardMaterial({
        color: 0xff3333,
        emissive: 0xaa1111,
        metalness: 0.7,
        roughness: 0.3
      });
      
      const redBase = new THREE.Mesh(redBaseGeometry, redBaseMaterial);
      redBase.position.set(150, 0, 0);
      redBase.isEnemyBase = true;
      redBase.team = 'red';
      redBase.health = 150000;
      scene.add(redBase);
      bases.push(redBase);
    }
    
    // Create player
    function createPlayer() {
      player = new THREE.Object3D();
      player.position.set(0, 5, 0);
      scene.add(player);
      
      // Add camera to player
      camera.position.set(0, 1.6, 0);
      player.add(camera);
      
      // Create a simple weapon model
      const weaponGeometry = new THREE.BoxGeometry(0.3, 0.3, 1.5);
      const weaponMaterial = new THREE.MeshPhongMaterial({
        color: playerTeam === 'blue' ? 0x3355ff : 0xff3333,
        emissive: playerTeam === 'blue' ? 0x1133aa : 0xaa1111,
        specular: 0xffffff,
        shininess: 30
      });
      
      const weapon = new THREE.Mesh(weaponGeometry, weaponMaterial);
      weapon.position.set(0.5, -0.5, -1);
      camera.add(weapon);
      
      // Set up player physics
      player.velocity = new THREE.Vector3();
      player.team = playerTeam;
      
      // Initialize player stats based on class
      updatePlayerClass();
    }
    
    // Create jetpack effect
    function createJetpackEffect() {
      const jetpackGeometry = new THREE.ConeGeometry(0.5, 2, 8);
      const jetpackMaterial = new THREE.MeshBasicMaterial({
        color: 0xffaa00,
        transparent: true,
        opacity: 0.8
      });
      
      jetpackEffect = new THREE.Mesh(jetpackGeometry, jetpackMaterial);
      jetpackEffect.rotation.x = Math.PI;
      jetpackEffect.position.set(0, -1, 0);
      jetpackEffect.visible = false;
      player.add(jetpackEffect);
    }
    
    // Spawn drones for both teams
    function spawnDrones() {
      // Clear existing drones
      drones.forEach(drone => scene.remove(drone));
      drones = [];
      
      // Spawn blue drones
      for (let i = 0; i < 9; i++) {
        const drone = createDrone('blue', i);
        drones.push(drone);
      }
      
      // Spawn red drones
      for (let i = 0; i < 10; i++) {
        const drone = createDrone('red', i);
        drones.push(drone);
      }
    }
    
    // Create a drone for a specific team
    function createDrone(team, index) {
      // Use a simple tetrahedron for drone geometry (triangle-based)
      const droneGeometry = new THREE.TetrahedronGeometry(1.5);
      const droneMaterial = new THREE.MeshPhongMaterial({
        color: team === 'blue' ? 0x3355ff : 0xff3333,
        emissive: team === 'blue' ? 0x1133aa : 0xaa1111,
        specular: 0xffffff,
        shininess: 50
      });
      
      const drone = new THREE.Mesh(droneGeometry, droneMaterial);
      drone.team = team;
      drone.health = 100;
      drone.lastFire = 0;
      
      // Position drones around their base
      const basePosition = team === 'blue' ? new THREE.Vector3(-150, 0, 0) : new THREE.Vector3(150, 0, 0);
      const angle = (index / (team === 'blue' ? 9 : 10)) * Math.PI * 2;
      const radius = 10 + Math.random() * 5;
      
      drone.position.set(
        basePosition.x + Math.cos(angle) * radius,
        5 + Math.random() * 5,
        basePosition.z + Math.sin(angle) * radius
      );
      
      scene.add(drone);
      return drone;
    }
    
    // Set up event listeners
    function setupEventListeners() {
      // Pointer lock for FPS controls
      document.addEventListener('click', () => {
        if (currentState === GameState.PLAYING && !isPointerLocked) {
          document.body.requestPointerLock = document.body.requestPointerLock ||
                                            document.body.mozRequestPointerLock ||
                                            document.body.webkitRequestPointerLock;
          document.body.requestPointerLock();
        }
      });
      
      document.addEventListener('pointerlockchange', () => {
        isPointerLocked = document.pointerLockElement === document.body;
        if (isPointerLocked) {
          // Hide cursor when pointer is locked
          document.body.style.cursor = 'none';
        } else {
          // Show cursor when pointer is unlocked
          document.body.style.cursor = 'auto';
        }
      });
      
      // Keyboard controls
      document.addEventListener('keydown', (e) => {
        if (currentState !== GameState.PLAYING) return;
        
        switch(e.code) {
          case 'KeyW':
            controls.forward = true;
            break;
          case 'KeyS':
            controls.backward = true;
            break;
          case 'KeyA':
            controls.left = true;
            break;
          case 'KeyD':
            controls.right = true;
            break;
          case 'Space':
            controls.fire = true;
            break;
          case 'ShiftLeft':
            if (playerClass === PlayerClass.SCOUT) {
              controls.jetpack = true;
            } else {
              controls.shield = true;
            }
            break;
          case 'ControlLeft':
            controls.crouch = true;
            break;
          case 'KeyR':
            controls.reload = true;
            break;
          case 'KeyQ':
            controls.leanLeft = true;
            break;
          case 'KeyE':
            controls.leanRight = true;
            break;
          case 'KeyF':
            if (playerClass === PlayerClass.MEDIC) {
              controls.heal = true;
            }
            break;
          case 'Escape':
            toggleMenu();
            break;
        }
      });
      
      document.addEventListener('keyup', (e) => {
        switch(e.code) {
          case 'KeyW':
            controls.forward = false;
            break;
          case 'KeyS':
            controls.backward = false;
            break;
          case 'KeyA':
            controls.left = false;
            break;
          case 'KeyD':
            controls.right = false;
            break;
          case 'Space':
            controls.fire = false;
            break;
          case 'ShiftLeft':
            controls.jetpack = false;
            controls.shield = false;
            break;
          case 'ControlLeft':
            controls.crouch = false;
            break;
          case 'KeyR':
            controls.reload = false;
            break;
          case 'KeyQ':
            controls.leanLeft = false;
            break;
          case 'KeyE':
            controls.leanRight = false;
            break;
          case 'KeyF':
            controls.heal = false;
            break;
        }
      });
      
      // Mouse controls
      document.addEventListener('mousedown', (e) => {
        if (currentState === GameState.PLAYING) {
          if (e.button === 0) {
            if (playerClass === PlayerClass.SCOUT) {
              controls.scan = true;
              performScan();
            } else {
              controls.fire = true;
            }
          } else if (e.button === 2) {
            if (playerClass === PlayerClass.SCOUT) {
              controls.scope = true;
              toggleScope(true);
            } else if (playerClass === PlayerClass.ASSAULT) {
              // Assault class shotgun blast
              fireShotgun();
            }
          }
        }
      });
      
      document.addEventListener('mouseup', (e) => {
        if (e.button === 0) {
          controls.scan = false;
          controls.fire = false;
        } else if (e.button === 2) {
          controls.scope = false;
          toggleScope(false);
        }
      });
      
      // Prevent right-click menu
      document.addEventListener('contextmenu', (e) => {
        if (currentState === GameState.PLAYING) {
          e.preventDefault();
        }
      });
      
      document.addEventListener('mousemove', (e) => {
        if (currentState === GameState.PLAYING && isPointerLocked) {
          // Rotate player based on mouse movement
          player.rotation.y -= e.movementX * mouseSensitivity;
          camera.rotation.x -= e.movementY * mouseSensitivity * (invertYAxis ? -1 : 1);
          
          // Limit vertical look
          camera.rotation.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, camera.rotation.x));
        }
      });
      
      // UI button event listeners
      document.getElementById('startBtn').addEventListener('click', startGame);
      
      document.getElementById('statsBtn').addEventListener('click', () => {
        updateStatsDisplay();
        document.getElementById('statsPanel').style.display = 'block';
      });
      
      document.getElementById('settingsBtn').addEventListener('click', () => {
        document.getElementById('settingsPanel').style.display = 'block';
      });
      
      document.getElementById('quitBtn').addEventListener('click', () => {
        window.close();
      });
      
      document.getElementById('closeSettings').addEventListener('click', () => {
        document.getElementById('settingsPanel').style.display = 'none';
      });
      
      document.getElementById('closeStats').addEventListener('click', () => {
        document.getElementById('statsPanel').style.display = 'none';
      });
      
      // Team selection
      document.getElementById('blueTeamBtn').addEventListener('click', () => {
        playerTeam = 'blue';
        document.getElementById('blueTeamBtn').classList.add('active');
        document.getElementById('redTeamBtn').classList.remove('active');
        if (player) {
          player.team = 'blue';
          updatePlayerWeaponColor();
        }
        
        // Adjust drone counts for team balance
        spawnDrones();
        updateUIForTeam();
      });
      
      document.getElementById('redTeamBtn').addEventListener('click', () => {
        playerTeam = 'red';
        document.getElementById('redTeamBtn').classList.add('active');
        document.getElementById('blueTeamBtn').classList.remove('active');
        if (player) {
          player.team = 'red';
          updatePlayerWeaponColor();
        }
        
        // Adjust drone counts for team balance
        spawnDrones();
        updateUIForTeam();
      });
      
      // Class selection
      document.getElementById('scoutClassBtn').addEventListener('click', () => {
        selectClass(PlayerClass.SCOUT);
      });
      
      document.getElementById('assaultClassBtn').addEventListener('click', () => {
        selectClass(PlayerClass.ASSAULT);
      });
      
      document.getElementById('medicClassBtn').addEventListener('click', () => {
        selectClass(PlayerClass.MEDIC);
      });
      
      // Handle window resize
      window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });
    }
    
    // Select class
    function selectClass(cls) {
      playerClass = cls;
      
      // Update UI
      document.getElementById('scoutClassBtn').classList.toggle('active', cls === PlayerClass.SCOUT);
      document.getElementById('assaultClassBtn').classList.toggle('active', cls === PlayerClass.ASSAULT);
      document.getElementById('medicClassBtn').classList.toggle('active', cls === PlayerClass.MEDIC);
      
      // Update button labels based on class
      if (cls === PlayerClass.SCOUT) {
        document.getElementById('jumpBtnLabel').textContent = 'JETPACK';
        document.getElementById('fireBtnLabel').textContent = 'FIRE';
        document.getElementById('sprintBtnLabel').textContent = 'CROUCH';
      } else if (cls === PlayerClass.ASSAULT) {
        document.getElementById('jumpBtnLabel').textContent = 'SHIELD';
        document.getElementById('fireBtnLabel').textContent = 'FIRE';
        document.getElementById('sprintBtnLabel').textContent = 'CROUCH';
      } else if (cls === PlayerClass.MEDIC) {
        document.getElementById('jumpBtnLabel').textContent = 'SHIELD';
        document.getElementById('fireBtnLabel').textContent = 'FIRE';
        document.getElementById('sprintBtnLabel').textContent = 'HEAL';
      }
      
      // Update player if game is already running
      if (currentState === GameState.PLAYING) {
        updatePlayerClass();
      }
    }
    
    // Update player based on selected class
    function updatePlayerClass() {
      const config = classConfig[playerClass];
      
      // Update player stats
      player.health = config.health;
      player.shield = config.shield;
      player.energy = config.energy;
      player.jetpackFuel = config.jetpackFuel;
      player.ammo = config.primaryWeapon.ammo;
      player.maxAmmo = config.primaryWeapon.maxAmmo;
      
      // Update weapon HUD
      document.getElementById('weaponIcon').textContent = config.primaryWeapon.icon;
      updateAmmoDisplay();
      
      // Update health, shield, and energy bars
      updateStatusBars();
      
      // Show/hide jetpack fuel display
      document.getElementById('jetpackFuelDisplay').style.display = config.hasJetpack ? 'block' : 'none';
    }
    
    // Update player weapon color based on team
    function updatePlayerWeaponColor() {
      const weapon = camera.children.find(child => child.isMesh);
      if (weapon) {
        weapon.material.color.set(playerTeam === 'blue' ? 0x3355ff : 0xff3333);
        weapon.material.emissive.set(playerTeam === 'blue' ? 0x1133aa : 0xaa1111);
      }
    }
    
    // Update UI colors based on team
    function updateUIForTeam() {
      const color = playerTeam === 'blue' ? '#55f' : '#f55';
      
      // Update UI elements
      const elementsToUpdate = [
        '.combat-stats',
        '.currency-display',
        '.ui-btn',
        '.menu-btn',
        '.health-fill',
        '.shield-fill'
      ];
      
      elementsToUpdate.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
          el.style.borderColor = color;
          el.style.color = color;
        });
      });
    }
    
    // Toggle scope view
    function toggleScope(enable) {
      if (enable) {
        document.getElementById('scopeOverlay').style.display = 'block';
        document.getElementById('scopeCrosshair').style.display = 'block';
        document.getElementById('crosshair').style.display = 'none';
        camera.fov = 30; // Zoom in
      } else {
        document.getElementById('scopeOverlay').style.display = 'none';
        document.getElementById('scopeCrosshair').style.display = 'none';
        document.getElementById('crosshair').style.display = 'block';
        camera.fov = fov; // Reset to normal FOV
      }
      camera.updateProjectionMatrix();
    }
    
    // Perform scan to detect enemies
    function performScan() {
      const scanEffect = document.getElementById('scanEffect');
      scanEffect.classList.add('active');
      
      setTimeout(() => {
        scanEffect.classList.remove('active');
      }, 2000);
      
      // Find all enemy drones and mark them
      const enemyDrones = drones.filter(drone => drone.team !== playerTeam);
      
      enemyDrones.forEach(drone => {
        // Create a temporary indicator for the drone
        const indicator = document.createElement('div');
        indicator.className = 'target-indicator';
        document.getElementById('gameContainer').appendChild(indicator);
        
        // Update position function
        const updateIndicatorPos = () => {
          const vector = drone.position.clone();
          vector.project(camera);
          
          if (vector.z > 0 && vector.z <= 1) {
            indicator.style.opacity = '1';
            indicator.style.left = `${(vector.x * 0.5 + 0.5) * window.innerWidth}px`;
            indicator.style.top = `${(-vector.y * 0.5 + 0.5) * window.innerHeight}px`;
          } else {
            indicator.style.opacity = '0';
          }
        };
        
        // Update position on each frame
        const indicatorId = setInterval(updateIndicatorPos, 100);
        
        // Remove indicator after 5 seconds
        setTimeout(() => {
          clearInterval(indicatorId);
          if (indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
          }
        }, 5000);
      });
      
      showConsoleMessage("Scan complete. Enemy positions revealed.");
    }
    
    // Start the game
    function startGame() {
      currentState = GameState.PLAYING;
      document.getElementById('mainMenu').style.display = 'none';
      
      // Initialize the game if not already initialized
      if (!scene) {
        init();
      } else {
        // Reset game state
        resetGame();
      }
      
      // Request pointer lock
      document.body.requestPointerLock = document.body.requestPointerLock ||
                                        document.body.mozRequestPointerLock ||
                                        document.body.webkitRequestPointerLock;
      document.body.requestPointerLock();
      
      // Update weapon HUD
      updateWeaponHUD();
      
      showConsoleMessage("GalaxyCraft: Red vs Blue engaged! Eliminate all enemies!");
    }
    
    // Reset game state
    function resetGame() {
      // Reset player position
      player.position.set(0, 5, 0);
      player.velocity.set(0, 0, 0);
      
      // Reset player stats based on class
      updatePlayerClass();
      
      // Reset bases
      bases.forEach(base => {
        base.health = 150000;
      });
      
      // Update base health display
      document.getElementById('blueBaseHealth').textContent = '100%';
      document.getElementById('redBaseHealth').textContent = '100%';
      
      // Respawn drones
      spawnDrones();
      
      // Clear projectiles
      projectiles.forEach(projectile => scene.remove(projectile));
      projectiles = [];
    }
    
    // Toggle menu
    function toggleMenu() {
      if (currentState === GameState.MENU) {
        currentState = GameState.PLAYING;
        document.getElementById('mainMenu').style.display = 'none';
      } else {
        currentState = GameState.MENU;
        document.getElementById('mainMenu').style.display = 'flex';
        
        // Exit pointer lock when menu is opened
        if (isPointerLocked) {
          document.exitPointerLock();
          isPointerLocked = false;
        }
      }
    }
    
    // Show console message
    function showConsoleMessage(message) {
      const consoleMessage = document.getElementById('consoleMessage');
      consoleMessage.textContent = message;
      consoleMessage.classList.add('active');
      
      setTimeout(() => {
        consoleMessage.classList.remove('active');
      }, 3000);
    }
    
    // Update weapon HUD
    function updateWeaponHUD() {
      const weapon = classConfig[playerClass].primaryWeapon;
      document.getElementById('weaponIcon').textContent = weapon.icon;
      updateAmmoDisplay();
    }
    
    // Update ammo display
    function updateAmmoDisplay() {
      document.getElementById('ammoDisplay').textContent = `${player.ammo}/${player.maxAmmo}`;
    }
    
    // Update status bars
    function updateStatusBars() {
      const config = classConfig[playerClass];
      const healthPercent = (player.health / config.health) * 100;
      const shieldPercent = (player.shield / config.shield) * 100;
      const energyPercent = (player.energy / config.energy) * 100;
      const fuelPercent = (player.jetpackFuel / config.jetpackFuelMax) * 100;
      
      document.getElementById('healthFill').style.width = `${healthPercent}%`;
      document.getElementById('shieldFill').style.width = `${shieldPercent}%`;
      document.getElementById('energyFill').style.width = `${energyPercent}%`;
      document.getElementById('fuelFill').style.width = `${fuelPercent}%`;
    }
    
    // Fire weapon
    function fireWeapon() {
      const config = classConfig[playerClass];
      const weapon = config.primaryWeapon;
      const now = Date.now();
      
      if (now - player.lastFire < weapon.fireRate * 1000) return;
      
      if (player.ammo <= 0) {
        // Auto reload if out of ammo
        reloadWeapon();
        return;
      }
      
      player.ammo--;
      player.lastFire = now;
      updateAmmoDisplay();
      
      // Create raycast for shooting
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(), camera);
      
      // Check for drone hits
      const droneTargets = drones.filter(drone => drone.team !== playerTeam);
      const intersects = raycaster.intersectObjects(droneTargets);
      
      if (intersects.length > 0) {
        const drone = intersects[0].object;
        
        // Show hit marker
        const hitMarker = document.getElementById('hitMarker');
        hitMarker.classList.add('active');
        setTimeout(() => hitMarker.classList.remove('active'), 300);
        
        // Damage drone
        drone.health -= weapon.damage;
        
        if (drone.health <= 0) {
          // Drone destroyed
          scene.remove(drone);
          drones = drones.filter(d => d !== drone);
          
          // Update score
          player.score += 100;
          player.kills++;
          document.getElementById('combatScore').textContent = player.score;
          document.getElementById('enemyCount').textContent = drones.filter(d => d.team !== playerTeam).length;
          
          showConsoleMessage("Target eliminated!");
        }
      }
      
      // Check for base hits
      const baseTargets = bases.filter(base => base.team !== playerTeam);
      const baseIntersects = raycaster.intersectObjects(baseTargets);
      
      if (baseIntersects.length > 0) {
        const base = baseIntersects[0].object;
        
        // Show hit marker
        const hitMarker = document.getElementById('hitMarker');
        hitMarker.classList.add('active');
        setTimeout(() => hitMarker.classList.remove('active'), 300);
        
        // Damage base
        base.health -= weapon.damage;
        
        // Update base health display
        if (base.team === 'blue') {
          const healthPercent = Math.max(0, (base.health / 150000) * 100);
          document.getElementById('blueBaseHealth').textContent = `${Math.round(healthPercent)}%`;
        } else {
          const healthPercent = Math.max(0, (base.health / 150000) * 100);
          document.getElementById('redBaseHealth').textContent = `${Math.round(healthPercent)}%`;
        }
        
        showConsoleMessage("Base hit!");
      }
      
      // Create muzzle flash effect
      const muzzleFlash = new THREE.PointLight(0xffaa00, 1, 10);
      muzzleFlash.position.set(0.5, -0.5, -1.5);
      camera.add(muzzleFlash);
      
      setTimeout(() => {
        camera.remove(muzzleFlash);
      }, 50);
    }
    
    // Fire shotgun (assault class special)
    function fireShotgun() {
      const config = classConfig[PlayerClass.ASSAULT];
      const weapon = config.primaryWeapon;
      const now = Date.now();
      
      if (now - player.lastFire < weapon.fireRate * 1000) return;
      
      if (player.ammo <= 0) {
        // Auto reload if out of ammo
        reloadWeapon();
        return;
      }
      
      player.ammo--;
      player.lastFire = now;
      updateAmmoDisplay();
      
      // Create multiple raycasts for shotgun spread
      for (let i = 0; i < 8; i++) {
        const raycaster = new THREE.Raycaster();
        const spread = 0.1;
        const direction = new THREE.Vector2(
          (Math.random() - 0.5) * spread,
          (Math.random() - 0.5) * spread
        );
        raycaster.setFromCamera(direction, camera);
        
        // Check for drone hits
        const droneTargets = drones.filter(drone => drone.team !== playerTeam);
        const intersects = raycaster.intersectObjects(droneTargets);
        
        if (intersects.length > 0) {
          const drone = intersects[0].object;
          
          // Show hit marker
          const hitMarker = document.getElementById('hitMarker');
          hitMarker.classList.add('active');
          setTimeout(() => hitMarker.classList.remove('active'), 300);
          
          // Damage drone (shotgun does 3x damage)
          drone.health -= weapon.damage * 3;
          
          if (drone.health <= 0) {
            // Drone destroyed
            scene.remove(drone);
            drones = drones.filter(d => d !== drone);
            
            // Update score
            player.score += 100;
            player.kills++;
            document.getElementById('combatScore').textContent = player.score;
            document.getElementById('enemyCount').textContent = drones.filter(d => d.team !== playerTeam).length;
            
            showConsoleMessage("Target eliminated with shotgun!");
          }
        }
      }
      
      // Create muzzle flash effect
      const muzzleFlash = new THREE.PointLight(0xffaa00, 2, 15);
      muzzleFlash.position.set(0.5, -0.5, -1.5);
      camera.add(muzzleFlash);
      
      setTimeout(() => {
        camera.remove(muzzleFlash);
      }, 50);
    }
    
    // Reload weapon
    function reloadWeapon() {
      const config = classConfig[playerClass];
      const weapon = config.primaryWeapon;
      
      if (player.isReloading) return;
      
      const ammoNeeded = weapon.ammo - player.ammo;
      
      if (ammoNeeded > 0 && player.maxAmmo > 0) {
        player.isReloading = true;
        showConsoleMessage("Reloading...");
        
        setTimeout(() => {
          const ammoToReload = Math.min(ammoNeeded, player.maxAmmo);
          player.ammo += ammoToReload;
          player.maxAmmo -= ammoToReload;
          player.isReloading = false;
          
          updateAmmoDisplay();
          showConsoleMessage("Reload complete!");
        }, weapon.reloadTime * 1000);
      }
    }
    
    // Update player movement
    function updatePlayer(deltaTime) {
      const config = classConfig[playerClass];
      
      // Calculate movement speed
      let speed = controls.sprint ? config.sprintSpeed : config.speed;
      
      // Handle crouching
      if (controls.crouch) {
        speed *= 0.6;
      }
      
      // Apply gravity if not using jetpack
      if (!controls.jetpack) {
        player.velocity.y -= 30 * deltaTime;
        
        // Regenerate jetpack fuel when not in use
        if (config.hasJetpack) {
          player.jetpackFuel = Math.min(
            config.jetpackFuelMax,
            player.jetpackFuel + config.jetpackFuelRegen * deltaTime
          );
        }
        
        // Hide jetpack effect
        if (jetpackEffect) jetpackEffect.visible = false;
      } else {
        // Use jetpack if fuel available
        if (player.jetpackFuel > 0) {
          player.velocity.y = config.jetpackStrength * deltaTime;
          player.jetpackFuel = Math.max(
            0,
            player.jetpackFuel - config.jetpackFuelConsumption * deltaTime
          );
          
          // Show jetpack effect
          if (jetpackEffect) jetpackEffect.visible = true;
        } else {
          // Out of fuel, fall naturally
          controls.jetpack = false;
          player.velocity.y -= 30 * deltaTime;
          
          // Hide jetpack effect
          if (jetpackEffect) jetpackEffect.visible = false;
        }
      }
      
      // Apply movement based on controls
      if (controls.forward) {
        player.translateZ(-speed * deltaTime);
      }
      
      if (controls.backward) {
        player.translateZ(speed * deltaTime * 0.7);
      }
      
      if (controls.left) {
        player.translateX(-speed * deltaTime * 0.8);
      }
      
      if (controls.right) {
        player.translateX(speed * deltaTime * 0.8);
      }
      
      // Apply velocity
      player.position.add(player.velocity.clone().multiplyScalar(deltaTime));
      
      // Check ground collision
      const raycaster = new THREE.Raycaster(
        player.position.clone(),
        new THREE.Vector3(0, -1, 0),
        0,
        1.1
      );
      
      const groundIntersections = raycaster.intersectObjects(terrain);
      controls.isGrounded = groundIntersections.length > 0;
      
      if (controls.isGrounded) {
        player.position.y = groundIntersections[0].point.y + 1.1;
        player.velocity.y = 0;
      }
      
      // Keep player within bounds
      player.position.x = Math.max(-190, Math.min(190, player.position.x));
      player.position.z = Math.max(-90, Math.min(90, player.position.z));
      
      // Handle firing
      if (controls.fire) {
        fireWeapon();
      }
      
      // Handle reloading
      if (controls.reload) {
        reloadWeapon();
        controls.reload = false;
      }
      
      // Handle shield ability
      if (controls.shield && config.hasShield && player.energy >= config.specialAbility.energyCost) {
        activateShield();
      }
      
      // Handle heal ability
      if (controls.heal && config.hasHeal && player.energy >= config.specialAbility.energyCost) {
        activateHeal();
      }
      
      // Regenerate energy
      player.energy = Math.min(config.energy, player.energy + config.energyRegen * deltaTime);
      
      // Update status bars
      updateStatusBars();
    }
    
    // Activate shield ability
    function activateShield() {
      const config = classConfig[playerClass];
      const now = Date.now();
      
      if (now - player.lastAbilityTime > config.specialAbility.cooldown * 1000) {
        player.energy -= config.specialAbility.energyCost;
        player.lastAbilityTime = now;
        
        // Show shield effect
        showShieldEffect();
        
        // Double shield temporarily
        player.shield = Math.min(config.shield * 2, player.shield + config.shield);
        
        showConsoleMessage("Energy shield activated!");
        
        // Reset shield after duration
        setTimeout(() => {
          player.shield = config.shield;
          updateStatusBars();
        }, config.specialAbility.duration * 1000);
      } else {
        showConsoleMessage("Shield ability on cooldown");
      }
    }
    
    // Activate heal ability
    function activateHeal() {
      const config = classConfig[playerClass];
      const now = Date.now();
      
      if (now - player.lastAbilityTime > config.specialAbility.cooldown * 1000) {
        player.energy -= config.specialAbility.energyCost;
        player.lastAbilityTime = now;
        
        // Show heal effect
        showHealEffect();
        
        // Heal player
        player.health = Math.min(config.health, player.health + 50);
        
        showConsoleMessage("Healing activated!");
        
        updateStatusBars();
      } else {
        showConsoleMessage("Heal ability on cooldown");
      }
    }
    
    // Update drones
    function updateDrones(deltaTime) {
      for (let i = drones.length - 1; i >= 0; i--) {
        const drone = drones[i];
        
        // Find targets (enemy drones, enemy base, or player if on opposite team)
        let targets = [];
        
        // Enemy drones
        targets = targets.concat(drones.filter(d => d.team !== drone.team));
        
        // Enemy base
        const enemyBase = bases.find(b => b.team !== drone.team);
        if (enemyBase) targets.push(enemyBase);
        
        // Player if on opposite team
        if (player && player.team !== drone.team) {
          targets.push(player);
        }
        
        // Find closest target
        let closestTarget = null;
        let closestDistance = Infinity;
        
        for (const target of targets) {
          const distance = drone.position.distanceTo(target.position);
          if (distance < closestDistance) {
            closestDistance = distance;
            closestTarget = target;
          }
        }
        
        if (closestTarget) {
          // Move toward target
          const direction = new THREE.Vector3().subVectors(closestTarget.position, drone.position).normalize();
          drone.position.add(direction.multiplyScalar(6 * deltaTime));
          
          // Keep drones at a certain height
          drone.position.y = 5 + Math.sin(Date.now() * 0.002 + i) * 2;
          
          // Face the direction of movement
          drone.lookAt(drone.position.clone().add(direction));
        }
      }
    }
    
    // Update HUD
    function updateHUD() {
      // Update health and shield bars
      updateStatusBars();
      
      // Update enemy count
      document.getElementById('enemyCount').textContent = drones.filter(d => d.team !== playerTeam).length;
      document.getElementById('combatScore').textContent = player.score;
      
      // Update base health
      const blueBase = bases.find(b => b.team === 'blue');
      const redBase = bases.find(b => b.team === 'red');
      
      if (blueBase) {
        const blueHealthPercent = Math.max(0, (blueBase.health / 150000) * 100);
        document.getElementById('blueBaseHealth').textContent = `${Math.round(blueHealthPercent)}%`;
      }
      
      if (redBase) {
        const redHealthPercent = Math.max(0, (redBase.health / 150000) * 100);
        document.getElementById('redBaseHealth').textContent = `${Math.round(redHealthPercent)}%`;
      }
      
      // Check for game over
      if (blueBase && blueBase.health <= 0) {
        showConsoleMessage("Red team wins! Blue base destroyed.");
        currentState = GameState.MENU;
        document.getElementById('mainMenu').style.display = 'flex';
      } else if (redBase && redBase.health <= 0) {
        showConsoleMessage("Blue team wins! Red base destroyed.");
        currentState = GameState.MENU;
        document.getElementById('mainMenu').style.display = 'flex';
      }
    }
    
    // Update stats display
    function updateStatsDisplay() {
      document.getElementById('statKills').textContent = player.kills;
      document.getElementById('statDeaths').textContent = player.deaths;
      document.getElementById('statScore').textContent = player.score;
      document.getElementById('statDamage').textContent = player.damageDealt;
      
      const accuracy = player.shotsFired > 0 
        ? Math.round((player.shotsHit / player.shotsFired) * 100) 
        : 0;
      document.getElementById('statAccuracy').textContent = `${accuracy}%`;
    }
    
    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      
      const deltaTime = Math.min(clock.getDelta(), 0.033); // Cap at ~30fps
      
      if (currentState === GameState.PLAYING) {
        updatePlayer(deltaTime);
        updateDrones(deltaTime);
        updateHUD();
      }
      
      if (renderer && scene && camera) {
        renderer.render(scene, camera);
      }
    }
    
    // Initialize the game when the page loads
    window.addEventListener('load', function() {
      // Set up event listeners for UI
      document.getElementById('startBtn').addEventListener('click', startGame);
      
      // Create stars background
      createStars();
      
      // Show main menu
      document.getElementById('mainMenu').style.display = 'flex';
    });
  </script>
</body>
</html>
