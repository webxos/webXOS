class AuthManager:
    def __init__(self):
        self.tokens = {}

    def validate_token(self, token):
        if not token or token not in self.tokens:
            return False
        return self.tokens[token].get('expires_in', 0) > 0

    def store_token(self, token, data):
        self.tokens[token] = {
            'access_token': token,
            'vials': data.get('vials', []),
            'expires_in': data.get('expires_in', 3600)
        }

    def get_user_vials(self, token):
        return self.tokens.get(token, {}).get('vials', [])

auth_manager = AuthManager()
