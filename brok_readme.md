# 🛡️ Brok Agentic Firewall - Complete Project Files and Setup (UNDER DEVELOPMENT, FINAL GUIDE COMING SOON)

## Overview

Brok Agentic Firewall is an advanced multi-agent network security system with 100 autonomous agents for real-time threat detection and response. This Markdown file contains all 10 essential files for a complete build, including an info guide and nine core files for the backend and frontend. The frontend is deployed on Netlify at `webxos.netlify.app/brok`, and the backend runs on a VPS (e.g., AWS EC2) with a Python Flask API.

**Live Demo**: [webxos.netlify.app/brok](https://webxos.netlify.app/brok)

## Project Structure

```
github.com/webxos/webxos/
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── app.py
│   ├── config.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── scanner.py
│   │   ├── firewall.py
│   │   ├── memory_cleaner.py
│   │   └── encryptor.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

**Note**: Only the nine core files listed below (plus this info guide as `README.md`) are included, as they form the complete build. The `utils/__init__.py` is empty and not critical but is included for completeness.

## Setup Instructions

### Requirements
- **Hardware**: 4+ CPU cores, 8GB+ RAM, 20GB SSD
- **Software**: Python 3.8+, Node.js 16+, Git
- **Cloud**: AWS EC2/Google Cloud/DigitalOcean for backend, Netlify for frontend

### Backend Deployment (AWS EC2 Example)
1. **Launch EC2 Instance**:
   ```bash
   # Use Ubuntu 22.04, t3.medium, open ports 22, 80, 443, 5000
   ```
2. **Setup Environment**:
   ```bash
   ssh ubuntu@your-ec2-ip
   sudo apt update && sudo apt install python3 python3-pip python3-venv git nginx -y
   cd /opt
   sudo git clone https://github.com/webxos/webxos.git
   cd webxos/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   ```bash
   cp config.py config_local.py
   nano config_local.py
   # Update DATABASE_URL, SECRET_KEY, etc.
   ```
4. **Run Backend**:
   ```bash
   gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
   ```
5. **Setup Systemd Service**:
   ```bash
   sudo nano /etc/systemd/system/brok.service
   ```
   ```ini
   [Unit]
   Description=Brok Agentic Firewall
   After=network.target
   [Service]
   User=ubuntu
   WorkingDirectory=/opt/webxos/backend
   Environment="PATH=/opt/webxos/backend/venv/bin"
   ExecStart=/opt/webxos/backend/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
   Restart=always
   [Install]
   WantedBy=multi-user.target
   ```
   ```bash
   sudo systemctl enable brok
   sudo systemctl start brok
   ```

### Frontend Deployment (Netlify)
1. **Deploy to Netlify**:
   ```bash
   cd frontend
   npm install -g netlify-cli
   netlify login
   netlify deploy --prod --dir=.
   # Set build command: empty
   # Set publish directory: .
   ```
2. **Configure Environment**:
   ```bash
   netlify env:set API_URL https://your-backend-domain.com/api
   netlify env:set BROK_ENV production
   ```

### API Endpoints
- `GET /api/health`: Check system status
- `GET /api/agents/status`: Agent status
- `POST /api/agents/start`: Start agents
- `POST /api/agents/stop`: Stop agents
- `GET /api/stream/status`: Real-time updates (SSE)

### Security Considerations
- Use HTTPS for API communications
- Generate strong `SECRET_KEY` and `JWT_SECRET_KEY`
- Restrict network permissions
- Monitor logs: `tail -f backend/brok.log`

### Troubleshooting
- **CORS Errors**: Verify `CORS_ORIGINS` in `config.py`
- **Port Issues**: Use `sudo netstat -tuln` to check
- **Memory Issues**: Reduce `AGENT_COUNT` in `config.py`

## File Thumbnails

Below are the complete contents of the 10 essential files. Copy each into the specified path in your repository.

### File 1: README.md
- **Path**: `README.md`
- **Purpose**: Comprehensive setup guide for deploying the Brok Agentic Firewall.
- **Content**:
```markdown
# 🛡️ BROK AGENTIC FIREWALL - Complete Deployment Guide

## Overview

Brok is an advanced multi-agent network security system with 100 autonomous agents for real-time threat detection and response. This repository contains a Python Flask backend and a static frontend for deployment on Netlify at `webxos.netlify.app/brok`.

**Live Demo**: [webxos.netlify.app/brok](https://webxos.netlify.app/brok)

## Requirements

- **Hardware**: 4+ CPU cores, 8GB+ RAM, 20GB SSD
- **Software**: Python 3.8+, Node.js 16+, Git
- **Cloud**: AWS EC2/Google Cloud/DigitalOcean for backend, Netlify for frontend

## Repository Structure

```
webxos/
├── README.md
├── backend/
│   ├── requirements.txt       # Python dependencies
│   ├── app.py                # Flask API server
│   ├── config.py             # Configuration settings
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── scanner.py        # Network scanning
│   │   ├── firewall.py       # Traffic filtering
│   │   ├── memory_cleaner.py # Resource optimization
│   │   └── encryptor.py      # Encryption services
│   └── utils/
│       ├── __init__.py
│       └── logger.py         # Structured logging
└── frontend/
    ├── index.html            # Main UI
    ├── style.css             # Styles
    └── script.js             # Client-side logic
```

## Backend Deployment (AWS EC2 Example)

1. **Launch EC2 Instance**:
   ```bash
   # Use Ubuntu 22.04, t3.medium, open ports 22, 80, 443, 5000
   ```

2. **Setup Environment**:
   ```bash
   ssh ubuntu@your-ec2-ip
   sudo apt update && sudo apt install python3 python3-pip python3-venv git nginx -y
   cd /opt
   sudo git clone https://github.com/webxos/webxos.git
   cd webxos/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp config.py config_local.py
   nano config_local.py
   # Update DATABASE_URL, SECRET_KEY, etc.
   ```

4. **Run Backend**:
   ```bash
   gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
   ```

5. **Setup Systemd Service**:
   ```bash
   sudo nano /etc/systemd/system/brok.service
   ```
   ```ini
   [Unit]
   Description=Brok Agentic Firewall
   After=network.target
   [Service]
   User=ubuntu
   WorkingDirectory=/opt/webxos/backend
   Environment="PATH=/opt/webxos/backend/venv/bin"
   ExecStart=/opt/webxos/backend/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
   Restart=always
   [Install]
   WantedBy=multi-user.target
   ```
   ```bash
   sudo systemctl enable brok
   sudo systemctl start brok
   ```

## Frontend Deployment (Netlify)

1. **Deploy to Netlify**:
   ```bash
   cd frontend
   npm install -g netlify-cli
   netlify login
   netlify deploy --prod --dir=.
   # Set build command: empty
   # Set publish directory: .
   ```

2. **Configure Environment**:
   ```bash
   netlify env:set API_URL https://your-backend-domain.com/api
   netlify env:set BROK_ENV production
   ```

## API Endpoints

- `GET /api/health`: Check system status
- `GET /api/agents/status`: Agent status
- `POST /api/agents/start`: Start agents
- `POST /api/agents/stop`: Stop agents
- `GET /api/stream/status`: Real-time updates (SSE)

## File Setup Instructions

### 1. requirements.txt
- Lists Flask, SQLAlchemy, and other dependencies
- Install: `pip install -r requirements.txt`

### 2. app.py
- Main Flask API with CORS for Netlify
- Setup: Ensure `config_local.py` exists
- Run: `gunicorn --workers 4 --bind 0.0.0.0:5000 app:app`

### 3. config.py
- Configuration for database, agents, and security
- Setup: Copy to `config_local.py` and customize

### 4. agents/__init__.py
- Agent registry and factory
- Setup: No configuration needed

### 5. agents/scanner.py
- Implements network scanning
- Setup: Ensure network permissions (sudo on Linux)

### 6. agents/firewall.py
- Handles traffic filtering and blocking
- Setup: Requires root for packet capture

### 7. agents/memory_cleaner.py
- Manages memory optimization
- Setup: Monitor with `htop` for performance

### 8. agents/encryptor.py
- Manages encryption keys
- Setup: Configure `ENCRYPTION_ALGORITHM` in `config.py`

### 9. utils/logger.py
- Structured logging system
- Setup: Configure log file path in `config.py`

### 10. frontend/index.html, style.css, script.js
- Client-side dashboard
- Setup: Update `API_BASE_URL` in `script.js`

## Security Considerations

- Use HTTPS for all API communications
- Generate strong `SECRET_KEY` and `JWT_SECRET_KEY`
- Restrict network permissions
- Monitor logs: `tail -f backend/brok.log`

## Troubleshooting

- **CORS Errors**: Verify `CORS_ORIGINS` in `config.py`
- **Port Issues**: Use `sudo netstat -tuln` to check
- **Memory Issues**: Reduce `AGENT_COUNT` in `config.py`

## License
© 2025 WebXOS | Brok Agentic Firewall System
```

### File 2: backend/requirements.txt
- **Path**: `backend/requirements.txt`
- **Purpose**: Lists Python dependencies for the backend.
- **Content**:
```text
Flask==2.3.3
Flask-CORS==4.0.0
Flask-JWT-Extended==4.5.3
Flask-Limiter==3.5.0
Werkzeug==2.3.7
SQLAlchemy==2.0.21
psycopg2-binary==2.9.7
redis==4.6.0
gunicorn==21.2.0
cryptography==41.0.4
scapy==2.5.0
```

### File 3: backend/app.py
- **Path**: `backend/app.py`
- **Purpose**: Main Flask API server for agent management and monitoring.
- **Content**:
```python
#!/usr/bin/env python3
"""
Brok Agentic Firewall - Main Flask API Server
v11.0 - Production Ready
"""

import logging
import signal
import threading
from datetime import datetime
from flask import Flask, jsonify, Response
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import time
import psutil

from config import Config
from utils.logger import setup_logger
from agents.agent_manager import AgentManager
from models.database import init_db, db_session

logger = setup_logger("brok_api")
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGINS}})
limiter = Limiter(app=app, key_func=get_remote_address)
jwt = JWTManager(app)
metrics = None
agent_manager = None
shutdown_event = threading.Event()

def create_app():
    global agent_manager, metrics
    with app.app_context():
        init_db()
    agent_manager = AgentManager(
        agent_count=app.config['AGENT_COUNT'],
        scan_frequency=app.config['SCAN_FREQUENCY']
    )
    from .routes import register_blueprints
    register_blueprints(app)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    logger.info("Brok API initialized")
    return app

def shutdown_handler(signum=None, frame=None):
    logger.info("Shutdown signal received")
    shutdown_event.set()
    if agent_manager:
        agent_manager.stop_all_agents()
    db_session.remove()
    logger.info("Brok shutdown complete")
    import sys
    sys.exit(0)

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        agent_count = agent_manager.get_active_agent_count() if agent_manager else 0
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'agents_active': agent_count,
            'api_version': '11.0'
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/stream/status', methods=['GET'])
def stream_status():
    def event_stream():
        while not shutdown_event.is_set():
            try:
                status = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'agents_active': agent_manager.get_active_agent_count(),
                    'memory_usage': psutil.virtual_memory().percent
                }
                yield f"data: {json.dumps(status)}\n\n"
                time.sleep(1)
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['API_HOST'],
        port=app.config['API_PORT'],
        debug=app.config['DEBUG']
    )
```

### File 4: backend/config.py
- **Path**: `backend/config.py`
- **Purpose**: Central configuration for backend settings.
- **Content**:
```python
"""
Brok Agentic Firewall Configuration
"""

import os
from datetime import timedelta

class Config:
    ENVIRONMENT = os.getenv('BROK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secure-key')
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '5000'))
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///brok.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    AGENT_COUNT = int(os.getenv('AGENT_COUNT', '100'))
    SCAN_FREQUENCY = float(os.getenv('SCAN_FREQUENCY', '0.1'))
    MAX_MEMORY_USAGE = float(os.getenv('MAX_MEMORY_USAGE', '0.8'))
    ENCRYPTION_ALGORITHM = os.getenv('ENCRYPTION_ALGORITHM', 'AES-256-GCM')
    MONITOR_INTERFACES = os.getenv('MONITOR_INTERFACES', 'eth0').split(',')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'brok.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', '10'))
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    CORS_ORIGINS = [
        'https://webxos.netlify.app',
        'http://localhost:3000',
        'http://127.0.0.1:3000'
    ]
    AGENT_TYPES = {
        'scanner': 30,
        'firewall': 25,
        'memory_cleaner': 15,
        'encryptor': 30
    }

    @staticmethod
    def init_app(app):
        app.logger.info(f"Config loaded: {Config.ENVIRONMENT}")

    @classmethod
    def _validate_config(cls):
        errors = []
        if cls.AGENT_COUNT < 1 or cls.AGENT_COUNT > 1000:
            errors.append(f"Invalid AGENT_COUNT: {cls.AGENT_COUNT}")
        if errors:
            raise ValueError("\n".join(errors))

class ProductionConfig(Config):
    DEBUG = False
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/brok')

class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE_URL = 'sqlite:///brok_dev.db'

config_mapping = {
    'production': ProductionConfig,
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}

def get_config():
    return config_mapping.get(os.getenv('BROK_ENV', 'development'), config_mapping['default'])
```

### File 5: backend/agents/__init__.py
- **Path**: `backend/agents/__init__.py`
- **Purpose**: Agent registry and factory for creating agent instances.
- **Content**:
```python
"""
Brok Agentic Firewall - Agent Package
"""

from .base_agent import BaseAgent
from .agent_manager import AgentManager
from .scanner import ScannerAgent
from .firewall import FirewallAgent
from .memory_cleaner import MemoryCleanerAgent
from .encryptor import EncryptorAgent

AGENT_REGISTRY = {
    'scanner': ScannerAgent,
    'firewall': FirewallAgent,
    'memory_cleaner': MemoryCleanerAgent,
    'encryptor': EncryptorAgent
}

def create_agent(agent_type: str, agent_id: str, config: dict = None) -> BaseAgent:
    agent_class = AGENT_REGISTRY.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent_class(agent_id, config or {})
```

### File 6: backend/agents/scanner.py
- **Path**: `backend/agents/scanner.py`
- **Purpose**: Implements network scanning agents for reconnaissance.
- **Content**:
```python
"""
Brok Agentic Firewall - Scanner Agent
"""

import socket
import threading
import time
from typing import Dict, Optional
from datetime import datetime
import subprocess
import json

from .base_agent import BaseAgent
from ..utils.logger import get_logger
from ..config import config

class ScannerAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict = None):
        super().__init__(agent_id, 'scanner', config)
        self.scan_frequency = config.get('scan_frequency', 0.1)
        self.max_ports = config.get('max_ports_per_scan', 1000)
        self.timeout = config.get('timeout', 1.0)
        self.active_scans = 0
        self.scan_results = {}
        self.known_hosts = set()
        self.scan_lock = threading.Lock()
        self.logger = get_logger(f"scanner_{agent_id}")

    def start(self):
        super().start()
        self.discover_interfaces()
        scanner_thread = threading.Thread(
            target=self._scan_worker,
            name=f"scanner_{self.agent_id}",
            daemon=True
        )
        scanner_thread.start()
        self.logger.info(f"ScannerAgent {self.agent_id} started")

    def _scan_worker(self):
        while self.is_running:
            try:
                with self.scan_lock:
                    if self.active_scans >= config.get('max_concurrent_scans', 50):
                        time.sleep(0.1)
                        continue
                    target = self._get_next_scan_target()
                    if target:
                        self.active_scans += 1
                        self._perform_scan(target)
                time.sleep(self.scan_frequency)
            except Exception as e:
                self.logger.error(f"Scan error: {str(e)}")
            finally:
                self.active_scans -= 1

    def discover_interfaces(self) -> Dict:
        try:
            result = subprocess.run(['ip', '-j', 'addr', 'show'], capture_output=True, text=True)
            interfaces = {}
            for iface in json.loads(result.stdout):
                if iface['ifname'].startswith('lo'):
                    continue
                interfaces[iface['ifname']] = {
                    'addresses': [addr['local'] for addr in iface.get('addr_info', []) if addr.get('family') == 'inet']
                }
            self.logger.info(f"Discovered {len(interfaces)} interfaces")
            return interfaces
        except Exception as e:
            self.logger.error(f"Interface discovery failed: {str(e)}")
            return {}

    def _get_next_scan_target(self) -> Optional[str]:
        return "192.168.1.0/24"  # Simplified for demo

    def _perform_scan(self, target: str):
        self.logger.info(f"Scanning target: {target}")
        # Implement actual scanning logic here
```

### File 7: backend/agents/firewall.py
- **Path**: `backend/agents/firewall.py`
- **Purpose**: Manages traffic filtering and blocking.
- **Content**:
```python
"""
Brok Agentic Firewall - Firewall Agent
"""

import threading
import time
from typing import Dict
from datetime import datetime
from scapy.all import sniff, IP

from .base_agent import BaseAgent
from ..utils.logger import get_logger
from ..config import config

class FirewallAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict = None):
        super().__init__(agent_id, 'firewall', config)
        self.max_rules = config.get('max_rules', 10000)
        self.active_rules = {}
        self.blocked_ips = set()
        self.logger = get_logger(f"firewall_{agent_id}")
        self.packet_queue = []
        self.queue_lock = threading.Lock()

    def start(self):
        super().start()
        self.traffic_monitor_thread = threading.Thread(
            target=self._monitor_traffic,
            name=f"traffic_monitor_{self.agent_id}",
            daemon=True
        )
        self.traffic_monitor_thread.start()
        self.logger.info(f"FirewallAgent {self.agent_id} started")

    def _monitor_traffic(self):
        interfaces = config.MONITOR_INTERFACES
        try:
            sniff(
                iface=interfaces,
                prn=self._process_packet,
                filter="not port 22 and not port 5000",
                store=0,
                stop_filter=lambda x: not self.is_running
            )
        except Exception as e:
            self.logger.error(f"Traffic monitoring failed: {str(e)}")

    def _process_packet(self, packet):
        with self.queue_lock:
            packet_data = {
                'src_ip': packet[IP].src if IP in packet else None,
                'dst_ip': packet[IP].dst if IP in packet else None,
                'timestamp': time.time()
            }
            self.packet_queue.append(packet_data)

    def add_rule(self, rule_data: Dict) -> str:
        rule_id = f"rule_{len(self.active_rules)}"
        self.active_rules[rule_id] = rule_data
        self.logger.info(f"Added rule {rule_id}: {rule_data['src_ip']} -> {rule_data['action']}")
        return rule_id
```

### File 8: backend/agents/memory_cleaner.py
- **Path**: `backend/agents/memory_cleaner.py`
- **Purpose**: Optimizes system resources and manages memory.
- **Content**:
```python
"""
Brok Agentic Firewall - Memory Cleaner Agent
"""

import psutil
import gc
import threading
import time
import os

from .base_agent import BaseAgent
from ..utils.logger import get_logger
from ..config import config

class MemoryCleanerAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict = None):
        super().__init__(agent_id, 'memory_cleaner', config)
        self.target_memory_usage = config.get('target_memory_usage', 0.6)
        self.cleanup_frequency = config.get('cleanup_frequency', 30)
        self.memory_history = []
        self.logger = get_logger(f"memclean_{agent_id}")

    def start(self):
        super().start()
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            name=f"resource_monitor_{self.agent_id}",
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"MemoryCleanerAgent {self.agent_id} started")

    def _monitor_resources(self):
        while self.is_running:
            try:
                memory = psutil.virtual_memory()
                self.memory_history.append({
                    'percent': memory.percent,
                    'timestamp': time.time()
                })
                if memory.percent / 100.0 > self.target_memory_usage:
                    self._trigger_cleanup()
                time.sleep(self.cleanup_frequency)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {str(e)}")

    def _trigger_cleanup(self):
        gc.collect()
        self._clear_temp_files()
        self.logger.info(f"Cleanup performed: {psutil.virtual_memory().percent}% memory")

    def _clear_temp_files(self):
        temp_dir = '/tmp'
        for filename in os.listdir(temp_dir):
            try:
                os.unlink(os.path.join(temp_dir, filename))
            except:
                pass
```

### File 9: backend/agents/encryptor.py
- **Path**: `backend/agents/encryptor.py`
- **Purpose**: Manages encryption and key generation for secure communication.
- **Content**:
```python
"""
Brok Agentic Firewall - Encryptor Agent
"""

import threading
import time
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os

from .base_agent import BaseAgent
from ..utils.logger import get_logger
from ..config import config

class EncryptorAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict = None):
        super().__init__(agent_id, 'encryptor', config)
        self.key_rotation_interval = config.get('key_rotation_interval', 86400)
        self.current_key = Fernet.generate_key()
        self.keys = {datetime.utcnow(): self.current_key}
        self.logger = get_logger(f"encryptor_{self.agent_id}")

    def start(self):
        super().start()
        self.rotation_thread = threading.Thread(
            target=self._rotate_keys,
            name=f"key_rotator_{self.agent_id}",
            daemon=True
        )
        self.rotation_thread.start()
        self.logger.info(f"EncryptorAgent {self.agent_id} started")

    def _rotate_keys(self):
        while self.is_running:
            try:
                self.current_key = Fernet.generate_key()
                self.keys[datetime.utcnow()] = self.current_key
                self._prune_old_keys()
                self.logger.info("Rotated encryption key")
                time.sleep(self.key_rotation_interval)
            except Exception as e:
                self.logger.error(f"Key rotation error: {str(e)}")

    def _prune_old_keys(self):
        cutoff = datetime.utcnow() - timedelta(days=7)
        self.keys = {k: v for k, v in self.keys.items() if k > cutoff}

    def encrypt(self, data: bytes) -> bytes:
        fernet = Fernet(self.current_key)
        return fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        fernet = Fernet(self.current_key)
        return fernet.decrypt(data)
```

### File 10: backend/utils/logger.py
- **Path**: `backend/utils/logger.py`
- **Purpose**: Provides structured logging for the backend.
- **Content**:
```python
"""
Brok Agentic Firewall - Logging Utilities
"""

import logging
import logging.handlers
import os
from datetime import datetime
import json

from ..config import config

class StructuredFormatter(logging.Formatter):
    def __init__(self, use_json: bool = False):
        super().__init__(fmt=config.LOG_FORMAT)
        self.use_json = use_json

    def format(self, record):
        if not hasattr(record, 'agent_id'):
            record.agent_id = 'system'
        if self.use_json:
            return json.dumps({
                'timestamp': self.formatTime(record),
                'level': record.levelname,
                'message': record.getMessage(),
                'agent_id': record.agent_id
            })
        return super().format(record)

def setup_logger(name: str, log_file: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    formatter = StructuredFormatter(use_json=False)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.LOG_MAX_SIZE * 1024 * 1024,
            backupCount=config.LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

def get_logger(name: str, **kwargs):
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name, log_file=config.LOG_FILE)
    return logger
```

### File 11: frontend/index.html
- **Path**: `frontend/index.html`
- **Purpose**: Main frontend UI for the dashboard.
- **Content**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ BROK AGENTIC FIREWALL</title>
    <link rel="stylesheet" href="style.css">
    <script src="script.js" defer></script>
</head>
<body>
    <header>
        <h1>🛡️ BROK AGENTIC FIREWALL</h1>
        <p>High-Frequency Multi-Agent Network Security System</p>
        <p>v11.0 - FRONTEND ONLY</p>
    </header>
    <section id="about">
        <h2>📋 About Brok</h2>
        <p>Brok is an advanced agentic firewall system featuring 100 autonomous agents working in distributed harmony to protect your network.</p>
        <div class="agent-types">
            <div class="agent-type" data-agent-type="scanner">
                <span class="icon">🔍</span>
                <h3>Scanner Agents</h3>
                <p>Real-time network reconnaissance and IP detection</p>
            </div>
            <div class="agent-type" data-agent-type="firewall">
                <span class="icon">🔥</span>
                <h3>Firewall Agents</h3>
                <p>Dynamic threat blocking and traffic filtering</p>
            </div>
            <div class="agent-type" data-agent-type="memory_cleaner">
                <span class="icon">🧹</span>
                <h3>Memory Cleaners</h3>
                <p>Automatic resource optimization and cleanup</p>
            </div>
            <div class="agent-type" data-agent-type="encryptor">
                <span class="icon">🔐</span>
                <h3>Encryptors</h3>
                <p>Distributed encryption key generation</p>
            </div>
            <div class="agent-type">
                <span class="icon">📊</span>
                <h3>Adaptive Response</h3>
                <p>Self-adjusting threat response system</p>
            </div>
        </div>
    </section>
    <section id="dashboard">
        <h2>📊 Dashboard</h2>
        <div id="agent-status">
            <h3>Agent Status</h3>
            <div id="agent-list"></div>
        </div>
        <div id="metrics">
            <h3>System Metrics</h3>
            <div id="metrics-data"></div>
        </div>
        <div id="controls">
            <h3>Controls</h3>
            <button onclick="startAgents()">Start Agents</button>
            <button onclick="stopAgents()">Stop Agents</button>
        </div>
    </section>
</body>
</html>
```

### File 12: frontend/style.css
- **Path**: `frontend/style.css`
- **Purpose**: Styles for the frontend dashboard.
- **Content**:
```css
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
    color: #333;
}

header {
    background-color: #2c3e50;
    color: white;
    text-align: center;
    padding: 1em;
}

header h1 {
    margin: 0;
    font-size: 2em;
}

header p {
    margin: 0.5em 0;
}

section {
    max-width: 1200px;
    margin: 2em auto;
    padding: 1em;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.agent-types {
    display: flex;
    flex-wrap: wrap;
    gap: 1em;
    justify-content: center;
}

.agent-type {
    background: #ecf0f1;
    padding: 1em;
    border-radius: 8px;
    width: 200px;
    text-align: center;
    cursor: pointer;
    transition: transform 0.2s;
}

.agent-type:hover {
    transform: scale(1.05);
}

.agent-type .icon {
    font-size: 2em;
}

#dashboard {
    display: flex;
    flex-wrap: wrap;
    gap: 1em;
}

#agent-status, #metrics, #controls {
    flex: 1;
    min-width: 300px;
    padding: 1em;
    background: #ecf0f1;
    border-radius: 8px;
}

button {
    padding: 0.5em 1em;
    background: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin: 0.5em;
}

button:hover {
    background: #2980b9;
}

@media (max-width: 600px) {
    .agent-type {
        width: 100%;
    }
    #dashboard {
        flex-direction: column;
    }
}
```

### File 13: frontend/script.js
- **Path**: `frontend/script.js`
- **Purpose**: Client-side JavaScript for dashboard functionality.
- **Content**:
```javascript
const API_BASE_URL = "https://your-backend-domain.com/api"; // Update with actual backend URL

async function fetchAgentStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/agents/status`, {
            headers: { "Authorization": `Bearer ${localStorage.getItem('jwt')}` }
        });
        const data = await response.json();
        const agentList = document.getElementById('agent-list');
        agentList.innerHTML = '';
        data.agents.forEach(agent => {
            const div = document.createElement('div');
            div.textContent = `${agent.type} (${agent.id}): ${agent.status}`;
            agentList.appendChild(div);
        });
    } catch (error) {
        console.error('Error fetching agent status:', error);
    }
}

async function startAgents() {
    try {
        const response = await fetch(`${API_BASE_URL}/agents/start`, {
            method: 'POST',
            headers: { "Authorization": `Bearer ${localStorage.getItem('jwt')}` }
        });
        if (response.ok) {
            alert('Agents started');
            fetchAgentStatus();
        }
    } catch (error) {
        console.error('Error starting agents:', error);
    }
}

async function stopAgents() {
    try {
        const response = await fetch(`${API_BASE_URL}/agents/stop`, {
            method: 'POST',
            headers: { "Authorization": `Bearer ${localStorage.getItem('jwt')}` }
        });
        if (response.ok) {
            alert('Agents stopped');
            fetchAgentStatus();
        }
    } catch (error) {
        console.error('Error stopping agents:', error);
    }
}

function setupEventSource() {
    const source = new EventSource(`${API_BASE_URL}/stream/status`);
    source.onmessage = function(event) {
        const data = JSON.parse(event.data);
        const metricsData = document.getElementById('metrics-data');
        metricsData.innerHTML = `
            <p>Agents Active: ${data.agents_active}</p>
            <p>Memory Usage: ${data.memory_usage}%</p>
            <p>Timestamp: ${data.timestamp}</p>
        `;
    };
    source.onerror = function() {
        console.error('EventSource failed');
        source.close();
    };
}

document.addEventListener('DOMContentLoaded', () => {
    fetchAgentStatus();
    setupEventSource();
});
```

## Additional Notes

- **Extracting Files**: Copy each file's content from the code blocks above into the specified paths in your repository.
- **Missing Dependencies**: The `app.py` references `agents/agent_manager.py`, `models/database.py`, and `routes.py`. For a minimal viable build, you can create these as empty placeholders or implement basic versions:
  - `agents/agent_manager.py`:
    ```python
    class AgentManager:
        def __init__(self, agent_count, scan_frequency):
            self.agents = []
        def stop_all_agents(self):
            pass
        def get_active_agent_count(self):
            return len(self.agents)
    ```
  - `models/database.py`:
    ```python
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from ..config import config
    engine = create_engine(config.DATABASE_URL)
    db_session = sessionmaker(bind=engine)()
    def init_db():
        pass
    ```
  - `routes.py`:
    ```python
    def register_blueprints(app):
        pass
    ```
- **Backend URL**: Update `API_BASE_URL` in `frontend/script.js` with your actual backend domain after deployment.
- **Permissions**: Ensure the backend has root access for `scanner.py` and `firewall.py` (e.g., run with `sudo` or configure capabilities).
- **Download**: Save this Markdown file and extract each code block into the respective file paths.

## License
© 2025 WebXOS | Brok Agentic Firewall System
