class OAuthConfig:
    def __init__(self):
        self.providers = {
            'mock': {
                'client_id': 'mock_client_id_789',
                'client_secret': 'mock_secret_abc',
                'redirect_uri': 'https://webxos.netlify.app/vial2/api/auth/callback',
                'scopes': ['openid', 'profile'],
                'endpoint': 'https://mock-oauth-provider.com'
            }
        }

    def get_provider_config(self, provider):
        return self.providers.get(provider, {})

    def validate_credentials(self, provider, code):
        config = self.get_provider_config(provider)
        return bool(config and code == 'test_code' and config.get('client_id') and config.get('client_secret'))

oauth_config = OAuthConfig()
