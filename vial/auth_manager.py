import uuid
import time

class AuthManager:
    def __init__(self):
        self.tokens = {}
        self.wallet = WebXOSWallet()
    
    def authenticate(self, client, device_id, session_id, network_id):
        if client == 'vial':
            token = str(uuid.uuid4())
            self.tokens[token] = {'device_id': device_id, 'session_id': session_id, 'network_id': network_id, 'timestamp': time.time()}
            self.wallet.address = str(uuid.uuid4())
            return token
        return None
    
    def verify_token(self, auth_header):
        if not auth_header or not auth_header.startswith('Bearer '):
            return False
        token = auth_header.split(' ')[1]
        return token in self.tokens
    
    def get_wallet_address(self):
        return self.wallet.address
