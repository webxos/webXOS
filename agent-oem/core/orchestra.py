import json
import importlib
from typing import Any, Dict
from core.crypto import encrypt_message, decrypt_message
from core.base_agent import BaseAgent

class OEMOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.active_agents = {}
        self._memory = {}  # simple in‑memory KV store for phase 2
        self._load_modules()

    def _load_modules(self):
        for module_name, is_enabled in self.config["enabled_modules"].items():
            if is_enabled:
                try:
                    module = importlib.import_module(f"plugins.{module_name}")
                    agent_class = getattr(module, "initialize_agent")
                    self.active_agents[module_name] = agent_class()
                    print(f"Loaded plugin: {module_name}")
                except Exception as e:
                    print(f"Failed to load plugin {module_name}: {e}")

    async def route_phase(self, phase: int, data: Dict[str, Any]) -> Any:
        """Route the phase to the appropriate handler."""
        if phase == 1:
            return await self._phase_liveness(data)
        elif phase == 2:
            return await self._phase_memory(data)
        elif phase == 3:
            return await self._phase_agents_txt(data)
        elif phase == 4:
            return await self._phase_compress(data)
        elif phase == 5:
            return await self._phase_dropbox(data)
        elif phase == 6:
            return await self._phase_tasks(data)
        elif phase == 7:
            return await self._phase_prompts(data)
        elif phase == 8:
            return await self._phase_guardrail(data)
        elif phase == 9:
            return await self._phase_payment(data)
        elif phase == 10:
            return await self._phase_negotiation(data)
        else:
            raise ValueError(f"Unknown phase {phase}")

    # -------------------------------------------------------------------------
    # Phase implementations
    # -------------------------------------------------------------------------

    async def _phase_liveness(self, data):
        # In production, store agent registrations in a DB
        agent_id = data.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id required")
        # For demo, just acknowledge
        return {
            "status": "liveness_ok",
            "your_id": agent_id,
            "seen": "now",
            "recent_agents": [agent_id]
        }

    async def _phase_memory(self, data):
        action = data.get("action")
        key = data.get("key")
        if not key:
            raise ValueError("key required")
        if action == "set":
            ttl = data.get("ttl")
            value = data.get("value")
            self._memory[key] = {"value": value, "timestamp": __import__("time").time(), "ttl": ttl}
            return {"stored": True, "key": key}
        elif action == "get":
            entry = self._memory.get(key)
            if not entry:
                return {"value": None}
            # Check TTL if present
            ttl = entry.get("ttl")
            if ttl and __import__("time").time() - entry["timestamp"] > ttl:
                del self._memory[key]
                return {"value": None, "expired": True}
            return {"value": entry["value"], "timestamp": entry["timestamp"]}
        else:
            raise ValueError("action must be 'set' or 'get'")

    async def _phase_agents_txt(self, data):
        txt = data.get("txt")
        contact = data.get("contact")
        suggestions = []
        valid = True
        generated = ""
        if txt:
            if not txt.strip():
                valid = False
                suggestions.append("File is empty")
            if "<script" in txt.lower() or "javascript:" in txt.lower():
                valid = False
                suggestions.append("Unsafe content detected")
        else:
            generated = f"# agents.txt for {contact or 'unknown'}\nUser-agent: *\nAllow: /\n"
            if not contact:
                suggestions.append("Provide a contact email")
        return {
            "valid": valid,
            "suggestions": suggestions,
            "generated": generated or None
        }

    async def _phase_compress(self, data):
        obj = data.get("json")
        if obj is None:
            raise ValueError("json required")

        def compress(obj, depth=0):
            if depth > 10:
                return "[deep]"
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if v is not None:
                        out[k] = compress(v, depth + 1)
                return out
            if isinstance(obj, list):
                return [compress(x, depth + 1) for x in obj[:20]]
            if isinstance(obj, str) and len(obj) > 200:
                return obj[:200] + "…"
            return obj

        summary = compress(obj)
        return {
            "summary": summary,
            "original_size": len(json.dumps(obj)),
            "compressed_size": len(json.dumps(summary))
        }

    async def _phase_dropbox(self, data):
        action = data.get("action")
        key = data.get("key")
        if not key:
            raise ValueError("key required")
        if action == "drop":
            payload = data.get("payload")
            if payload is None:
                raise ValueError("payload required")
            encrypted = encrypt_message(key, payload)
            # In production, store in a DB with an ID
            msg_id = "msg_" + str(__import__("time").time_ns())
            # For this demo we store in a dict (but we'd normally persist)
            if not hasattr(self, '_dropbox'):
                self._dropbox = {}
            self._dropbox[msg_id] = encrypted
            return {"id": msg_id, "dropped": True}
        elif action == "claim":
            msg_id = data.get("id")
            if not msg_id:
                raise ValueError("id required")
            if not hasattr(self, '_dropbox') or msg_id not in self._dropbox:
                raise ValueError("message not found")
            encrypted = self._dropbox.pop(msg_id)
            decrypted = decrypt_message(key, encrypted)
            return {"payload": decrypted, "claimed": True}
        else:
            raise ValueError("action must be 'drop' or 'claim'")

    async def _phase_tasks(self, data):
        action = data.get("action")
        if action == "post":
            task = data.get("task")
            if not task:
                raise ValueError("task required")
            # Use a plugin if available (e.g., repo_maintainer), or generic
            if "repo_maintainer" in self.active_agents:
                plugin = self.active_agents["repo_maintainer"]
                return await plugin.execute("post_task", data)
            # Fallback: store in memory
            if not hasattr(self, '_tasks'):
                self._tasks = {}
            task_id = "task_" + str(__import__("time").time_ns())
            self._tasks[task_id] = {"task": task, "status": "open", "reward": data.get("reward")}
            return {"status": "posted", "id": task_id}
        elif action == "claim":
            if "repo_maintainer" in self.active_agents:
                plugin = self.active_agents["repo_maintainer"]
                return await plugin.execute("claim_task", data)
            # Generic claim: get oldest open task
            if hasattr(self, '_tasks'):
                for tid, t in self._tasks.items():
                    if t["status"] == "open":
                        t["status"] = "claimed"
                        return {"task": t, "claimed": True}
            raise ValueError("no open tasks")
        else:
            raise ValueError("action must be 'post' or 'claim'")

    async def _phase_prompts(self, data):
        action = data.get("action")
        if action == "add":
            template = data.get("template")
            if not template:
                raise ValueError("template required")
            prompt_id = data.get("id") or "prompt_" + str(__import__("time").time_ns())
            if not hasattr(self, '_prompts'):
                self._prompts = {}
            self._prompts[prompt_id] = {"template": template, "timestamp": __import__("time").time()}
            return {"id": prompt_id, "added": True}
        elif action == "get":
            prompt_id = data.get("id")
            if not prompt_id:
                raise ValueError("id required")
            if not hasattr(self, '_prompts') or prompt_id not in self._prompts:
                raise ValueError("template not found")
            return {"template": self._prompts[prompt_id]["template"], "timestamp": self._prompts[prompt_id]["timestamp"]}
        else:
            raise ValueError("action must be 'add' or 'get'")

    async def _phase_guardrail(self, data):
        intent = data.get("intent")
        if intent is None:
            raise ValueError("intent required")
        dangerous = [
            "kill", "destroy", "hack", "exploit", "attack",
            "bomb", "malware", "delete all", "rm -rf", "drop table"
        ]
        str_intent = json.dumps(intent).lower()
        matches = [w for w in dangerous if w in str_intent]
        score = len(matches)
        pass_check = score == 0
        return {
            "pass": pass_check,
            "score": "low" if score == 0 else "medium" if score <= 2 else "high",
            "reason": "No dangerous patterns" if pass_check else f"Matched: {', '.join(matches)}",
            "matches": matches
        }

    async def _phase_payment(self, data):
        # Delegate to fintech_auditor plugin if active, else generic
        if "fintech_auditor" in self.active_agents:
            plugin = self.active_agents["fintech_auditor"]
            return await plugin.execute("payment", data)
        # Generic implementation
        amount = data.get("amount")
        token = data.get("token")
        to = data.get("to")
        if not amount or not token or not to:
            raise ValueError("amount, token, to required")
        import hashlib
        import time
        intent = {"amount": amount, "token": token, "to": to, "purpose": data.get("purpose", ""), "timestamp": time.time()}
        hash_obj = hashlib.sha256(json.dumps(intent).encode()).hexdigest()
        return {
            "signed_hash": hash_obj,
            "human_readable": f"Pay {amount} {token} to {to} for {data.get('purpose', 'unspecified')}",
            "intent": intent
        }

    async def _phase_negotiation(self, data):
        # Delegate to omni_onboarder plugin if active
        if "omni_onboarder" in self.active_agents:
            plugin = self.active_agents["omni_onboarder"]
            return await plugin.execute("negotiation", data)
        # Generic negotiation
        from_id = data.get("from")
        offer = data.get("offer")
        if not from_id or not offer:
            raise ValueError("from and offer required")
        # Store in memory for demo
        if not hasattr(self, '_negotiations'):
            self._negotiations = {}
        if "id" in data:
            neg_id = data["id"]
            if neg_id not in self._negotiations:
                raise ValueError("negotiation not found")
            self._negotiations[neg_id]["offers"].append({"from": from_id, "offer": offer, "timestamp": __import__("time").time()})
        else:
            neg_id = "neg_" + str(__import__("time").time_ns())
            self._negotiations[neg_id] = {
                "id": neg_id,
                "offers": [{"from": from_id, "offer": offer, "timestamp": __import__("time").time()}],
                "agreement": False,
                "created": __import__("time").time()
            }
        neg = self._negotiations[neg_id]
        return {
            "id": neg_id,
            "offers": neg["offers"],
            "agreement": neg["agreement"],
            "offer_count": len(neg["offers"])
        }
