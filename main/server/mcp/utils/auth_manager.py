import json
import os
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, storage_path='/tmp/auth_tokens.json'):
        self.storage_path = storage_path
        self.tokens = self._load_tokens()

    def _load_tokens(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_tokens(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.tokens, f)

    def validate_token(self, token):
        if not token or token not in self.tokens:
            return False
        token_data = self.tokens[token]
        expires_at = datetime.fromisoformat(token_data['expires_at'])
        return expires_at > datetime.now()

    def store_token(self, token, data):
        expires_at = datetime.now() + timedelta(seconds=data.get('expires_in', 3600))
        self.tokens[token] = {
            'access_token': token,
            'vials': data.get('vials', []),
            'expires_at': expires_at.isoformat()
        }
        self._save_tokens()

    def get_user_vials(self, token):
        return self.tokens.get(token, {}).get('vials', [])

auth_manager = AuthManager()
