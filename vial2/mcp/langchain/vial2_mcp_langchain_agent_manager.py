import asyncio
from typing import List

class AgentManager:
    def __init__(self):
        self.agents = {}

    async def add_agent(self, agent_id: str, agent):
        self.agents[agent_id] = agent
        await self.save_state()

    async def remove_agent(self, agent_id: str):
        self.agents.pop(agent_id, None)
        await self.save_state()

    async def get_agent(self, agent_id: str):
        return self.agents.get(agent_id)

    async def save_state(self):
        # Implement state persistence to NeonDB
        pass