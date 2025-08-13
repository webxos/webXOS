import aiohttp
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_key = None

    async def authenticate(self, user_id: str) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/auth", json={"userId": user_id}) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    data = await response.json()
                    self.api_key = data["apiKey"]
                    logger.info(f"Authenticated client for {user_id}")
                    return True
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Authentication error: {str(e)}\n")
            return False

    async def get_vials(self) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/vials", headers={"Authorization": f"Bearer {self.api_key}"}) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    data = await response.json()
                    logger.info(f"Retrieved {len(data['agents'])} vials")
                    return data["agents"]
        except Exception as e:
            logger.error(f"Vial retrieval error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Vial retrieval error: {str(e)}\n")
            return {}

    async def cashout(self, user_id: str, target_address: str, amount: float) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/wallet/cashout",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"transaction": {"amount": amount}, "wallet": {"target_address": target_address}}
                ) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    logger.info(f"Cashed out {amount} $WEBXOS to {target_address} for {user_id}")
                    return True
        except Exception as e:
            logger.error(f"Cashout error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Cashout error: {str(e)}\n")
            return False
