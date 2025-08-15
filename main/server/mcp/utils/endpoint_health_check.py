import requests
import time
from server.mcp.utils.logging_config import logger

class EndpointHealthCheck:
    def __init__(self, base_url='https://webxos.netlify.app/vial2/api'):
        self.base_url = base_url
        self.endpoints = ['troubleshoot', 'auth/oauth', 'health']

    def check_health(self):
        for endpoint in self.endpoints:
            url = f"{self.base_url}/{endpoint}"
            try:
                response = requests.get(url if endpoint == 'health' else f"{url}", timeout=5)
                if response.status_code == 200:
                    logger.info(f"Endpoint {url} is healthy")
                else:
                    logger.error(f"Endpoint {url} returned status {response.status_code}")
                    return False
            except requests.RequestException as e:
                logger.error(f"Health check failed for {url}: {str(e)}")
                return False
        return True

    def monitor(self, interval=60):
        while True:
            if not self.check_health():
                logger.warning("Health check failed. Retrying in 60 seconds.")
            time.sleep(interval)

if __name__ == "__main__":
    checker = EndpointHealthCheck()
    checker.monitor()
