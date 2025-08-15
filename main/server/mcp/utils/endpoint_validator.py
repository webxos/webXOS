import requests
import logging
from server.mcp.utils.logging_config import logger

class EndpointValidator:
    def __init__(self, base_url='https://webxos.netlify.app/vial2/api'):
        self.base_url = base_url

    def check_endpoint(self, endpoint):
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Endpoint {url} is available")
                return True
            logger.error(f"Endpoint {url} returned status {response.status_code}")
            return False
        except requests.RequestException as e:
            logger.error(f"Endpoint {url} check failed: {str(e)}")
            return False

    def validate_all(self):
        endpoints = ['troubleshoot', 'auth/oauth']
        return all(self.check_endpoint(endpoint) for endpoint in endpoints)

validator = EndpointValidator()
