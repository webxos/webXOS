import sqlite3
import uuid
import torch
import torch.nn as nn
from agent1 import Agent1
from agent2 import Agent2
from agent3 import Agent3
from agent4 import Agent4

class VialManager:
    def __init__(self, network_id):
        self.network_id = network_id
        self.vials = []
        self._init_vials()

    def _init_vials(self):
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('SELECT vial_id FROM vials WHERE network_id = ?', (self.network_id,))
        if not cursor.fetchall():
            agents = [Agent1(), Agent2(), Agent3(), Agent4()]
            for i, agent in enumerate(agents, 1):
                vial_id = f'vial{i}'
                code = agent.__class__.__name__
                cursor.execute('INSERT INTO vials (network_id, vial_id, status, code, code_length, is_python, webxos_hash, wallet_address, wallet_balance, tasks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                              (self.network_id, vial_id, 'stopped', code, len(code), True, str(uuid.uuid4()), str(uuid.uuid4()), 0.0, '[]'))
            conn.commit()
        cursor.execute('SELECT * FROM vials WHERE network_id = ?', (self.network_id,))
        rows = cursor.fetchall()
        for row in rows:
            self.vials.append({
                'id': row[1],
                'status': row[2],
                'code': row[3],
                'codeLength': row[4],
                'isPython': bool(row[5]),
                'webxosHash': row[6],
                'wallet': {'address': row[7], 'balance': row[8]},
                'tasks': eval(row[9])
            })
        conn.close()

    def train_vials(self, code, is_python):
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        for vial in self.vials:
            vial['status'] = 'running'
            vial['code'] = code
            vial['codeLength'] = len(code)
            vial['isPython'] = is_python
            vial['tasks'].append(f"task_{str(uuid.uuid4())}")
            cursor.execute('UPDATE vials SET status = ?, code = ?, code_length = ?, is_python = ?, tasks = ? WHERE network_id = ? AND vial_id = ?',
                          (vial['status'], vial['code'], vial['codeLength'], vial['isPython'], str(vial['tasks']), self.network_id, vial['id']))
        conn.commit()
        conn.close()

    def get_vials(self):
        return self.vials
