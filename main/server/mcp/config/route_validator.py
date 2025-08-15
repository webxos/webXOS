import requests
from server.mcp.utils.logging_config import logger

class RouteValidator:
    def __init__(self, base_url='https://webxos.netlify.app/vial2/api'):
        self.base_url = base_url
        self.endpoints = ['troubleshoot', 'auth/oauth']

    def validate_routes(self):
        for endpoint in self.endpoints:
            url = f"{self.base_url}/{endpoint}"
            try:
                response = requests.head(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Route {url} is valid")
                else:
                    logger.error(f"Route {url} returned status {response.status_code}")
                    return False
            except requests.RequestException as e:
                logger.error(f"Route validation failed for {url}: {str(e)}")
                return False
        return True

validator = RouteValidator()
