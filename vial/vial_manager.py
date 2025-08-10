import hashlib
import uuid
import torch
import logging
import sqlite3
from agent_host import MCPHost
from agent_client import MCPClient
from agent_server import MCPServer
from agent_protocol import MCPProtocol

class VialManager:
    def __init__(self):
        self.vials = {
            'vial1': {'agent': MCPHost(), 'role': 'host', 'status': 'stopped', 'webxosHash': str(uuid.uuid4()), 'wallet': {'address': None, 'balance': 0}, 'tasks': ['search_docs', 'read_emails']},
            'vial2': {'agent': MCPClient(), 'role': 'client', 'status': 'stopped', 'webxosHash': str(uuid.uuid4()), 'wallet': {'address': None, 'balance': 0}, 'tasks': ['send_gmails']},
            'vial3': {'agent': MCPServer(), 'role': 'server', 'status': 'stopped', 'webxosHash': str(uuid.uuid4()), 'wallet': {'address': None, 'balance': 0}, 'tasks': ['search_web']},
            'vial4': {'agent': MCPProtocol(), 'role': 'protocol', 'status': 'stopped', 'webxosHash': str(uuid.uuid4()), 'wallet': {'address': None, 'balance': 0}, 'tasks': []}
        }
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.db_path = '/app/vial.db'

    def generate_webxos_hash(self, code, network_id):
        try:
            return hashlib.sha256(f"{code}{network_id}".encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Hash generation error: {str(e)}")
            return str(uuid.uuid4())

    def train_vials(self, code, is_python, network_id):
        try:
            if not is_python:
                self.logger.error("Only Python code supported for training")
                return list(self.vials.values())
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            for vial_id, vial in self.vials.items():
                input_data = torch.rand(10, 10)
                vial['agent'].train(input_data, code)
                vial['status'] = 'running'
                vial['code'] = code
                vial['codeLength'] = len(code)
                vial['webxosHash'] = self.generate_webxos_hash(code, network_id)
                vial['wallet']['address'] = str(uuid.uuid4())
                vial['wallet']['balance'] = 0.0001  # Reward per vial
                c.execute("INSERT OR REPLACE INTO vials (id, status, code, codeLength, isPython, webxosHash, walletAddress, walletBalance, tasks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (vial_id, vial['status'], vial['code'], vial['codeLength'], is_python, vial['webxosHash'], vial['wallet']['address'], vial['wallet']['balance'], ','.join(vial['tasks'])))
                self.logger.info(f"Trained {vial_id} with role {vial['role']} and tasks {vial['tasks']}")
            conn.commit()
            conn.close()
            return [
                {
                    'id': vial_id,
                    'status': vial['status'],
                    'code': vial['code'],
                    'codeLength': vial['codeLength'],
                    'isPython': True,
                    'webxosHash': vial['webxosHash'],
                    'wallet': vial['wallet'],
                    'tasks': vial['tasks']
                } for vial_id, vial in self.vials.items()
            ]
        except Exception as e:
            self.logger.error(f"Train error: {str(e)}")
            return list(self.vials.values())

    def void_vials(self, network_id):
        try:
            for vial_id, vial in self.vials.items():
                vial['status'] = 'stopped'
                vial['code'] = ''
                vial['codeLength'] = 0
                vial['webxosHash'] = str(uuid.uuid4())
                vial['wallet'] = {'address': None, 'balance': 0}
                vial['tasks'] = vial['tasks']  # Retain tasks for template
                self.logger.info(f"Voided {vial_id} for network ID: {network_id}")
        except Exception as e:
            self.logger.error(f"Void error: {str(e)}")
