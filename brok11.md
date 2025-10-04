import socket
import time
import logging
import random
import string
import base64
from datetime import datetime
from collections import defaultdict, deque

# High-frequency logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Brok")

class PromptLog:
    def __init__(self, maxlen=100):
        self.logs = deque(maxlen=maxlen)  # Store up to 100 prompt logs

    def add_log(self, agent_id, role, event_type, action, details=""):
        """Add a prompt log entry."""
        log_entry = {
            "timestamp": datetime.now(),
            "agent_id": agent_id,
            "role": role,
            "event_type": event_type,  # Error or Intruder
            "action": action,
            "details": details
        }
        self.logs.append(log_entry)
        logger.info(f"Prompt Log: Agent {agent_id} (Role: {role}) - Event: {event_type}, Action: {action}, Details: {details}")

    def analyze_logs(self):
        """Analyze prompt logs to detect threat frequency and adapt."""
        if not self.logs:
            return 0, 0
        intruder_count = sum(1 for log in self.logs if log["event_type"] == "Intruder")
        error_count = sum(1 for log in self.logs if log["event_type"] == "Error")
        return intruder_count, error_count

class Agent:
    def __init__(self, id, role):
        self.id = id
        self.role = role  # Scanner, Firewall, MemoryCleaner, AttackDetector, Encryptor
        self.status = "standby"
        self.key_fragment = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.task = None
        self.prompt_log = None  # Set by AgenticNetwork

    def set_prompt_log(self, prompt_log):
        """Set the prompt log for the agent."""
        self.prompt_log = prompt_log

    def execute_task(self, task, *args):
        """Execute task if it matches the agent's role."""
        if not self.is_task_allowed(task):
            logger.debug(f"Agent {self.id} (Role: {self.role}) cannot execute task {task}")
            return None
        self.status = task
        self.task = task
        logger.info(f"Agent {self.id} (Role: {self.role}): Assigned task {task}")
        try:
            result = getattr(self, task, lambda *x: None)(*args)
            logger.info(f"Agent {self.id} (Role: {self.role}): Completed task {task}, Result: {result}")
            # Random feedback log
            if random.random() < 0.3:  # 30% chance to log feedback
                self.log_feedback()
            return result
        except Exception as e:
            self.alert_network("Error", f"Task {task} failed: {e}")
            return None

    def is_task_allowed(self, task):
        """Check if the task matches the agent's role."""
        role_tasks = {
            "Scanner": ["scan"],
            "Firewall": ["firewall"],
            "MemoryCleaner": ["clear_memory"],
            "AttackDetector": ["detect_attack"],
            "Encryptor": ["contribute_to_key"]
        }
        return task in role_tasks.get(self.role, [])

    def alert_network(self, event_type, details):
        """Alert the network of an error or intruder."""
        if self.prompt_log:
            action = "TriggerOverwhelmingResponse"
            self.prompt_log.add_log(self.id, self.role, event_type, action, details)
            logger.warning(f"Agent {self.id} (Role: {self.role}) alerted network: {event_type} - {details}")

    def scan(self, test_ip):
        """Agent-specific network scan."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.05)
            result = sock.connect_ex((test_ip, 53))
            sock.close()
            return test_ip, result
        except Exception as e:
            self.alert_network("Error", f"Scan error on {test_ip}: {e}")
            return test_ip, None

    def firewall(self, ip, connection_attempts, max_attempts, suspect_ips, blocked_ips, active_sockets):
        """Agent-specific firewall task."""
        try:
            if ip in suspect_ips or connection_attempts[ip] >= max_attempts:
                blocked_ips.add(ip)
                if ip in active_sockets:
                    active_sockets[ip].close()
                    del active_sockets[ip]
                return True
            return False
        except Exception as e:
            self.alert_network("Error", f"Firewall error on {ip}: {e}")
            return False

    def clear_memory(self, ip_logs, active_sockets, suspect_ips, connection_attempts):
        """Agent-specific memory cleanup."""
        try:
            if random.random() < 0.9:
                if ip_logs:
                    ip_logs.popleft()
                for ip in list(active_sockets.keys()):
                    active_sockets[ip].close()
                    del active_sockets[ip]
                if len(suspect_ips) > 10:
                    suspect_ips.pop()
                connection_attempts.clear()
            return True
        except Exception as e:
            self.alert_network("Error", f"Memory clear error: {e}")
            return False

    def detect_attack(self, ip_logs, connection_attempts, suspect_ips, attack_load):
        """Agent-specific attack detection."""
        try:
            if len(ip_logs) < 3:
                return False
            recent_logs = list(ip_logs)[-3:]
            ip_count = len(set(ip for _, ips in recent_logs for ip in ips))
            attempt_spike = sum(connection_attempts.values()) / max(1, len(connection_attempts))
            is_attack = ip_count > 3 or attempt_spike > 3 or len(suspect_ips) > 5 or attack_load > 10
            if is_attack:
                self.alert_network("Intruder", f"Attack detected: IPs={ip_count}, Avg Attempts={attempt_spike:.2f}, Load={attack_load}")
            return is_attack
        except Exception as e:
            self.alert_network("Error", f"Attack detection error: {e}")
            return False

    def contribute_to_key(self):
        """Contribute to encryption key."""
        try:
            fragment = list(self.key_fragment)
            random.shuffle(fragment)
            return ''.join(fragment)
        except Exception as e:
            self.alert_network("Error", f"Key contribution error: {e}")
            return ""

    def log_feedback(self):
        """Log random feedback with device stats."""
        cpu_usage = random.uniform(10, 90)  # Simulated CPU usage (%)
        mem_usage = random.uniform(20, 80)  # Simulated memory usage (%)
        net_load = random.randint(50, 500)  # Simulated network load (KB/s)
        feedback = f"Agent {self.id} (Role: {self.role}) Feedback: Task={self.task or 'None'}, " \
                   f"Status={self.status}, CPU={cpu_usage:.1f}%, Mem={mem_usage:.1f}%, Net={net_load}KB/s"
        logger.info(feedback)

class AgenticNetwork:
    def __init__(self):
        self.agents = []
        self.running = False
        self.agent_count = 0
        self.task_queue = deque()
        self.available_ids = list(range(1, 101))  # IDs 1–100
        self.roles = ["Scanner", "Firewall", "MemoryCleaner", "AttackDetector", "Encryptor"]
        self.target_role_count = 20  # Aim for ~20 agents per role
        self.prompt_log = PromptLog()
        self.threat_level = 0  # Tracks threat frequency for adaptation

    def start(self):
        """Start the agentic network with 100 agents and balanced roles."""
        if not self.running:
            logger.info("Starting agentic network...")
            self.running = True
            role_counts = {role: 0 for role in self.roles}
            while self.agent_count < 100 and self.running:
                self.agent_count += 1
                # Prefer roles with fewer agents, adjust for threats
                available_roles = [r for r in self.roles if role_counts[r] < self.target_role_count]
                if self.threat_level > 5:  # Increase AttackDetector/Firewall for high threats
                    available_roles = ["AttackDetector", "Firewall"] if available_roles else self.roles
                elif not available_roles:
                    available_roles = self.roles
                role = random.choice(available_roles)
                role_counts[role] += 1
                new_id = random.choice(self.available_ids)
                self.available_ids.remove(new_id)
                agent = Agent(new_id, role)
                agent.set_prompt_log(self.prompt_log)
                self.agents.append(agent)
                logger.info(f"Agent {new_id} (Role: {role}) created. Total agents: {self.agent_count}")
                time.sleep(0.01)
            logger.info("Agentic network is ready!")
        else:
            logger.info("Network already running.")

    def stop(self):
        """Stop the agentic network."""
        if self.running:
            logger.info("Stopping agentic network...")
            self.running = False
            self.agents.clear()
            self.agent_count = 0
            self.available_ids = list(range(1, 101))
            self.prompt_log.logs.clear()
        else:
            logger.info("Network is not running.")

    def regenerate_agent(self, old_agent):
        """Regenerate an agent with a new ID and role."""
        old_id = old_agent.id
        self.available_ids.append(old_id)
        self.agents.remove(old_agent)
        self.agent_count -= 1
        # Assign new role, considering threat level
        role_counts = {role: sum(1 for a in self.agents if a.role == role) for role in self.roles}
        available_roles = [r for r in self.roles if role_counts.get(r, 0) < self.target_role_count]
        if self.threat_level > 5:
            available_roles = ["AttackDetector", "Firewall"] if available_roles else self.roles
        elif not available_roles:
            available_roles = self.roles
        new_role = random.choice(available_roles)
        new_id = random.choice(self.available_ids)
        self.available_ids.remove(new_id)
        new_agent = Agent(new_id, new_role)
        new_agent.set_prompt_log(self.prompt_log)
        self.agents.append(new_agent)
        self.agent_count += 1
        logger.info(f"Agent {old_id} regenerated as Agent {new_id} (Role: {new_role})")
        return new_agent

    def assign_tasks(self, task_type, *args):
        """Assign tasks to agents based on their roles and regenerate them."""
        if not self.running or self.agent_count < 100:
            logger.warning("Network not fully initialized or not running.")
            return []
        # Select agents with the appropriate role
        role_for_task = {
            "scan": "Scanner",
            "firewall": "Firewall",
            "clear_memory": "MemoryCleaner",
            "detect_attack": "AttackDetector",
            "contribute_to_key": "Encryptor"
        }.get(task_type)
        if not role_for_task:
            logger.warning(f"No role defined for task {task_type}")
            return []
        eligible_agents = [agent for agent in self.agents if agent.role == role_for_task]
        if not eligible_agents:
            logger.warning(f"No agents available for task {task_type} (Role: {role_for_task})")
            return []
        random.shuffle(eligible_agents)
        # Adjust number of agents based on device stats and threat level
        max_agents = 10 if self.threat_level < 5 else 20  # More agents for high threats
        cpu_usage = random.uniform(10, 90)  # Simulate device load
        if cpu_usage > 80:  # Reduce agents if device is overloaded
            max_agents = max(5, max_agents // 2)
        results = []
        agents_to_regenerate = []
        for agent in eligible_agents[:min(len(eligible_agents), max_agents)]:
            result = agent.execute_task(task_type, *args)
            if result is not None:
                results.append(result)
            agents_to_regenerate.append(agent)
        # Regenerate agents after task completion
        for agent in agents_to_regenerate:
            self.regenerate_agent(agent)
        return results

    def overwhelming_response(self, event_type, details):
        """Trigger an overwhelming response to a threat."""
        logger.warning(f"Overwhelming response triggered: {event_type} - {details}")
        # Assign tasks to multiple agents based on their roles
        tasks = [
            ("scan", ["8.8.8.8", "1.1.1.1", "4.2.2.2"]),  # Intensify scanning
            ("firewall", ["0.0.0.0", defaultdict(int), 2, set(), set(), {}]),  # Aggressive blocking
            ("detect_attack", [deque(maxlen=50), defaultdict(int), set(), 0]),  # Heighten detection
            ("contribute_to_key", [])  # Prepare encryption key
        ]
        for task_type, args in tasks:
            self.assign_tasks(task_type, *args)

    def generate_encryption_key(self):
        """Generate a distributed encryption key."""
        if not self.running or self.agent_count < 100:
            logger.warning("Network not fully initialized or not running.")
            return None
        logger.info("Agents forming encryption key...")
        combined_key = ""
        results = self.assign_tasks("contribute_to_key")
        for fragment in results:
            if fragment:
                combined_key += fragment
        combined_key = combined_key[:32] if len(combined_key) >= 32 else combined_key.ljust(32, '0')
        logger.info("Encryption key generated across Encryptor agents.")
        return combined_key

    def adapt_network(self, cpu_usage, mem_usage):
        """Adapt network parameters based on prompt log and device stats."""
        intruder_count, error_count = self.prompt_log.analyze_logs()
        self.threat_level = intruder_count + error_count // 2  # Weighted threat level
        logger.info(f"Network adapting: Intruders={intruder_count}, Errors={error_count}, ThreatLevel={self.threat_level}")
        # Adjust backoff and attempts based on threat level
        if self.threat_level > 5:
            self.target_role_count = 25  # Increase AttackDetector/Firewall agents
        else:
            self.target_role_count = 20
        # Reduce activity if device is overloaded
        if cpu_usage > 80 or mem_usage > 80:
            logger.info("High device load detected, reducing agent activity.")
            return 0.5  # Reduce backoff factor
        return 1.0 if self.threat_level < 5 else 1.5  # Increase backoff for threats

class Brok:
    def __init__(self):
        self.scan_interval = 0.1  # 10Hz base frequency
        self.running = True
        self.scan_count = 0
        self.ip_logs = deque(maxlen=50)
        self.blocked_ips = set()
        self.connection_attempts = defaultdict(int)
        self.active_sockets = {}
        self.max_sockets = 50
        self.max_attempts_per_ip = 2
        self.backoff_factor = 1.0
        self.max_backoff = 10.0
        self.suspect_ips = set()
        self.last_reset = time.time()
        self.attack_load = 0
        self.network = AgenticNetwork()

    def shutdown(self):
        self.running = False
        self.network.stop()
        for sock in list(self.active_sockets.values()):
            try:
                sock.close()
            except:
                pass
        self.active_sockets.clear()
        logger.info("Brok: Shutdown completed.")

    def reset_brok(self):
        logger.debug("Resetting Brok...")
        self.ip_logs.clear()
        for sock in list(self.active_sockets.values()):
            try:
                sock.close()
            except:
                pass
        self.active_sockets.clear()
        self.scan_count += 1
        self.backoff_factor = max(0.5, self.backoff_factor * 0.9)
        self.last_reset = time.time()
        logger.debug("Reset complete.")

    def check_system_health(self):
        """Rapid health check."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.close()
            elapsed = time.time() - self.last_reset
            if elapsed > 60:
                logger.warning("Reset overdue, forcing reset.")
                return False
            return True
        except OSError as e:
            self.network.prompt_log.add_log(0, "System", "Error", "HealthCheckFailed", str(e))
            logger.error(f"Health check failed: {e}")
            return False

    def scan_network(self):
        start_time = time.time()
        external_ips = set()

        if not self.check_system_health():
            self.backoff_factor *= 2
            return external_ips, 0.0

        logger.debug("Starting distributed scan...")
        test_ips = ['8.8.8.8', '1.1.1.1', '4.2.2.2']
        for test_ip in test_ips:
            if test_ip in self.blocked_ips or self.connection_attempts[test_ip] >= self.max_attempts_per_ip:
                continue
            results = self.network.assign_tasks("scan", test_ip)
            for ip, result in results:
                if result == 0:
                    external_ips.add(ip)
                    self.connection_attempts[ip] += 1
                elif result in (111, 113):
                    self.connection_attempts[ip] += 1
                    self.suspect_ips.add(ip)
                elif result is None:
                    self.suspect_ips.add(ip)

        self.ip_logs.append((datetime.now(), list(external_ips)))
        elapsed = (time.time() - start_time) * 1000
        self.attack_load = max(0, self.attack_load - 0.1) if elapsed < 50 else self.attack_load + 1
        logger.debug(f"Distributed scan completed in {elapsed:.2f}ms")
        return external_ips, elapsed

    def simple_firewall(self):
        start_time = time.time()
        blocked_count = 0

        logger.debug("Starting distributed firewall...")
        for ip in list(self.connection_attempts.keys()):
            results = self.network.assign_tasks("firewall", ip, self.connection_attempts, 
                                              self.max_attempts_per_ip, self.suspect_ips, 
                                              self.blocked_ips, self.active_sockets)
            blocked_count += sum(1 for result in results if result)

        if self.ip_logs:
            recent_ips = self.ip_logs[-1][1]
            if len(self.active_sockets) >= self.max_sockets * 0.8 or self.attack_load > 5:
                logger.debug("Sandbox near full or attack detected, emptying...")
                for ip in list(self.active_sockets.keys()):
                    self.active_sockets[ip].close()
                    del self.active_sockets[ip]
                self.attack_load += 2
            elif len(self.active_sockets) < self.max_sockets:
                for ip in recent_ips:
                    if ip not in self.blocked_ips and ip not in self.active_sockets:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.05)
                        try:
                            sock.connect((ip, 53))
                            self.active_sockets[ip] = sock
                            self.connection_attempts[ip] += 1
                        except socket.timeout:
                            sock.close()
                            self.suspect_ips.add(ip)
                        except Exception as e:
                            self.network.prompt_log.add_log(0, "System", "Error", "SandboxError", str(e))
                            sock.close()
                            self.suspect_ips.add(ip)

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Distributed firewall completed in {elapsed:.2f}ms")
        return blocked_count, elapsed

    def clear_memory(self):
        start_time = time.time()
        success = all(self.network.assign_tasks("clear_memory", self.ip_logs, self.active_sockets, 
                                              self.suspect_ips, self.connection_attempts))
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Distributed memory clear {'successful' if success else 'failed'} in {elapsed:.2f}ms")
        return success, elapsed

    def detect_attack(self):
        """Distributed attack detection."""
        results = self.network.assign_tasks("detect_attack", self.ip_logs, self.connection_attempts, 
                                          self.suspect_ips, self.attack_load)
        attack_detected = any(results)
        if attack_detected:
            self.network.overwhelming_response("Intruder", "Attack detected by agents")
        return attack_detected

    def encrypt_data(self, data):
        """Encrypt data using agent-generated key."""
        key = self.network.generate_encryption_key()
        if not key:
            return None
        key_bytes = key.encode('utf-8')
        msg_bytes = data.encode('utf-8')
        encrypted = bytearray()
        for i in range(len(msg_bytes)):
            encrypted.append(msg_bytes[i] ^ key_bytes[i % len(key_bytes)])
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_data(self, encrypted, key=None):
        """Decrypt data using agent-generated key."""
        if not key:
            key = self.network.generate_encryption_key()
        if not key:
            return None
        key_bytes = key.encode('utf-8')
        enc_bytes = base64.b64decode(encrypted)
        decrypted = bytearray()
        for i in range(len(enc_bytes)):
            decrypted.append(enc_bytes[i] ^ key_bytes[i % len(key_bytes)])
        return decrypted.decode('utf-8')

    def report_stats(self):
        logger.debug("Reporting stats...")
        try:
            # Simulate device stats
            cpu_usage = random.uniform(10, 90)
            mem_usage = random.uniform(20, 80)
            # Adapt network based on logs and device stats
            self.backoff_factor *= self.network.adapt_network(cpu_usage, mem_usage)
            ext_ips, net_ms = self.scan_network()
            blocked, fw_ms = self.simple_firewall()
            mem_success, mem_time = self.clear_memory()
            attack_detected = self.detect_attack()
            logger.info(f"Scan #{self.scan_count + 1}: IPs={len(ext_ips)}, Blocked={blocked}, "
                        f"Time={net_ms + fw_ms:.2f}ms, Attack={attack_detected}")
            if attack_detected:
                self.backoff_factor = min(self.backoff_factor * 1.5, 5)
                # Encrypt critical logs during attack
                log_data = f"Scan #{self.scan_count + 1}: Attack detected"
                encrypted_log = self.encrypt_data(log_data)
                logger.info(f"Encrypted log: {encrypted_log}")
            elif self.attack_load < 1:
                self.backoff_factor = max(0.5, self.backoff_factor * 0.95)
        except Exception as e:
            self.network.prompt_log.add_log(0, "System", "Error", "ReportError", str(e))
            logger.error(f"Report error: {e}")
            self.backoff_factor *= 1.5

    def run_brok(self):
        logger.info("Brok: Starting High-Frequency Agentic Firewall...")
        self.network.start()
        retries = 0
        while self.running:
            try:
                start_cycle = time.time()
                logger.debug("Starting cycle...")
                self.reset_brok()
                self.report_stats()
                elapsed = time.time() - start_cycle
                adjusted_interval = self.scan_interval * self.backoff_factor
                sleep_time = max(0, adjusted_interval - elapsed)
                logger.debug(f"Cycle took {elapsed:.3f}s, sleeping {sleep_time:.3f}s")
                time.sleep(min(sleep_time, self.max_backoff))
                retries = 0
                self.attack_load = max(0, self.attack_load - 0.2)
            except KeyboardInterrupt:
                self.shutdown()
                break
            except Exception as e:
                self.network.prompt_log.add_log(0, "System", "Error", "RunError", str(e))
                logger.error(f"Run error: {e}")
                retries += 1
                backoff = min(self.scan_interval * (1.5 ** retries), self.max_backoff)
                self.backoff_factor = min(self.backoff_factor * 1.2, 5)
                time.sleep(backoff)

def main():
    brok = Brok()
    brok.run_brok()

if __name__ == "__main__":
    main()