import importlib
import yaml
from typing import Dict, Any

class OEMOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.active_agents = {}
        self._load_modules()

    def _load_modules(self):
        for module_name, is_enabled in self.config["enabled_modules"].items():
            if is_enabled:
                try:
                    module = importlib.import_module(f"plugins.{module_name}")
                    agent_class = getattr(module, "initialize_agent")
                    self.active_agents[module_name] = agent_class()
                except Exception as e:
                    print(f"Failed to load plugin {module_name}: {e}")

    async def route_phase(self, phase: int, data: Dict[str, Any]) -> Any:
        """
        Maps the 10 protocol phases to internal actions.
        Some phases are handled directly (memory, liveness, etc.),
        others are delegated to the appropriate plugin.
        """
        # Phase 1: Liveness – manage agents
        if phase == 1:
            return await self._phase_liveness(data)
        # Phase 2: Key-Value Memory (in‑memory store for demo – could use Redis)
        elif phase == 2:
            return await self._phase_memory(data)
        # Phase 3: agents.txt validator
        elif phase == 3:
            return await self._phase_agents_txt(data)
        # Phase 4: JSON semantic compressor
        elif phase == 4:
            return await self._phase_compress(data)
        # Phase 5: Encrypted dropbox – simple AES‑GCM (same as frontend)
        elif phase == 5:
            return await self._phase_dropbox(data)
        # Phase 6: Task FIFO – delegate to relevant plugin if needed
        elif phase == 6:
            return await self._phase_tasks(data)
        # Phase 7: Prompt registry
        elif phase == 7:
            return await self._phase_prompts(data)
        # Phase 8: Action Guardrail – can be used by all plugins
        elif phase == 8:
            return await self._phase_guardrail(data)
        # Phase 9: Payment intent – delegate to fintech plugin if active
        elif phase == 9:
            return await self._phase_payment(data)
        # Phase 10: Negotiation – delegate to omni_onboarder or generic
        elif phase == 10:
            return await self._phase_negotiation(data)
        else:
            raise ValueError(f"Unknown phase {phase}")

    # --------------------------------------------------------------------
    # Phase implementations (simplified for brevity – expand as needed)
    # --------------------------------------------------------------------
    async def _phase_liveness(self, data):
        # In production, store agent registrations in a DB
        return {"status": "liveness_ok", "message": f"Agent {data.get('agent_id')} seen"}

    async def _phase_memory(self, data):
        # Use a simple in‑memory dict
        if not hasattr(self, '_memory'):
            self._memory = {}
        action = data.get("action")
        key = data.get("key")
        if action == "set":
            self._memory[key] = data.get("value")
            return {"stored": True, "key": key}
        elif action == "get":
            return {"value": self._memory.get(key)}
        else:
            raise ValueError("action must be 'set' or 'get'")

    async def _phase_agents_txt(self, data):
        # Generate or validate
        return {"valid": True, "suggestions": [], "generated": "# agents.txt for OEM backend"}

    async def _phase_compress(self, data):
        # Simple compression
        import json
        obj = data.get("json")
        def compress(obj, depth=0):
            if depth > 10: return "[deep]"
            if isinstance(obj, dict):
                return {k: compress(v, depth+1) for k,v in obj.items() if v is not None}
            if isinstance(obj, list):
                return [compress(x, depth+1) for x in obj[:20]]
            if isinstance(obj, str) and len(obj) > 200:
                return obj[:200] + "…"
            return obj
        summary = compress(obj)
        return {"summary": summary, "original_size": len(json.dumps(obj)), "compressed_size": len(json.dumps(summary))}

    async def _phase_dropbox(self, data):
        # Use crypto (same as frontend) – we'll reuse the same functions
        from core.crypto import encrypt_message, decrypt_message
        action = data.get("action")
        key = data.get("key")
        if action == "drop":
            encrypted = await encrypt_message(key, data["payload"])
            return {"id": "dropbox-id", "dropped": True}
        elif action == "claim":
            # In reality we'd store messages; here we return a dummy
            return {"payload": "dummy_decrypted", "claimed": True}
        else:
            raise ValueError("action must be 'drop' or 'claim'")

    async def _phase_tasks(self, data):
        # Delegate to appropriate plugin if task type matches
        # For now, just return a dummy
        return {"status": "posted", "id": "task-123"}

    async def _phase_prompts(self, data):
        # Store/retrieve prompts
        return {"template": "You are a helpful assistant."}

    async def _phase_guardrail(self, data):
        intent = data.get("intent")
        dangerous = ["kill","destroy","hack","exploit","attack","bomb","malware","delete all","rm -rf","drop table"]
        matches = [w for w in dangerous if w in json.dumps(intent).lower()]
        score = len(matches)
        return {
            "pass": score == 0,
            "score": "low" if score == 0 else "medium" if score <= 2 else "high",
            "reason": "No dangerous patterns" if score == 0 else f"Matched: {', '.join(matches)}",
            "matches": matches
        }

    async def _phase_payment(self, data):
        # Use fintech_auditor plugin if enabled, else generic
        if "fintech_auditor" in self.active_agents:
            plugin = self.active_agents["fintech_auditor"]
            return await plugin.execute("payment", data)
        return {"signed_hash": "0x123", "human_readable": f"Pay {data.get('amount')} {data.get('token')} to {data.get('to')}"}

    async def _phase_negotiation(self, data):
        # Use omni_onboarder if enabled
        if "omni_onboarder" in self.active_agents:
            plugin = self.active_agents["omni_onboarder"]
            return await plugin.execute("negotiation", data)
        return {"offers": [{"from": data.get("from"), "offer": data.get("offer")}], "agreement": False}
