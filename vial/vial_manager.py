import torch
import json
from agent1 import Agent1
from agent2 import Agent2
from agent3 import Agent3
from agent4 import Agent4
from quantum_simulator import QuantumSimulator
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from vector_container import VectorContainer

class VialManager:
    def __init__(self, network_id):
        self.network_id = network_id
        self.agents = {
            'vial1': Agent1(),
            'vial2': Agent2(),
            'vial3': Agent3(),
            'vial4': Agent4()
        }
        self.simulator = QuantumSimulator()
        self.langchain_runnable = RunnableLambda(self.process_langchain_input)
        self.vector_container = VectorContainer()

    def train_vials(self, code, isPython):
        if "## WEBXOS Tokenization Tag:" not in code:
            raise ValueError("Invalid .md: Missing WEBXOS Tokenization Tag")
        chunks = self.vector_container.chunk_md(code)
        try:
            data = {}
            for chunk in chunks:
                if '<!-- Metadata:' in chunk:
                    meta_str = chunk.split('<!-- Metadata: ')[1].split(' -->')[0]
                    meta = json.loads(meta_str)
                if 'Vial Data' in chunk:
                    data.update(json.loads(chunk.split('## Vial Data')[1]))
            for vial_id, agent in self.agents.items():
                input_tensor = torch.tensor(data.get(vial_id, [0.0] * 10), dtype=torch.float32)
                output = agent(input_tensor)
                self.simulator.update_state(vial_id, output.item())
        except json.JSONDecodeError:
            raise ValueError("Invalid .md format: must contain JSON-like data for vials")

    def get_vials(self):
        return {
            vial_id: {
                'output': agent(torch.randn(10)).item(),
                'quantum_state': self.simulator.get_state(vial_id)
            } for vial_id, agent in self.agents.items()
        }

    def process_langchain_input(self, input_message):
        if isinstance(input_message, HumanMessage):
            code = input_message.content
            self.train_vials(code, isPython=False)
            return {"status": "processed", "vials": self.get_vials()}
        return {"error": "Invalid input for LangChain"}

# [xaiartifact: v1.7]
