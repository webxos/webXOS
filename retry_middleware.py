import requests
import time
import logging
from functools import wraps

# Setup logging
logging.basicConfig(filename='db/errorlog.md', level=logging.INFO, format='## [%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def retry_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 3
        delay = 1
        for attempt in range(retries):
            try:
                response = func(*args, **kwargs)
                if response.status_code == 200:
                    logger.info(f"Request to {args[0]} succeeded")
                    return response
                else:
                    logger.warning(f"Request to {args[0]} failed with status {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"Request to {args[0]} failed: {str(e)}")
                if attempt == retries - 1:
                    raise
                time.sleep(delay * (2 ** attempt))
        raise Exception(f"Request to {args[0]} failed after {retries} attempts")
    return wrapper

@retry_request
def make_request(url, method='GET', json=None, headers=None):
    return requests.request(method, url, json=json, headers=headers, timeout=5)
