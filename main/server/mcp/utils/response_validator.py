import requests
from server.mcp.utils.logging_config import logger

class ResponseValidator:
    def __init__(self, base_url='https://webxos.netlify.app/vial2/api'):
        self.base_url = base_url

    def validate_response(self, endpoint, method='POST', data=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.request(method, url, headers=headers, json=data, timeout=5)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if 'application/json' not in content_type:
                    logger.error(f"Endpoint {url} returned non-JSON: {content_type}")
                    return False
                return True
            logger.error(f"Endpoint {url} returned status {response.status_code}: {response.text}")
            return False
        except requests.RequestException as e:
            logger.error(f"Validation failed for {url}: {str(e)}")
            return False

validator = ResponseValidator()
