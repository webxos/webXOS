import json
import time
import importlib
from typing import Any, Dict, Optional, List, Union
from core.crypto import encrypt_message, decrypt_message
from core.base_agent import BaseAgent


class OEMOrchestrator:
    """
    Central orchestrator for the Agent Grounding protocol.
    Dynamically loads enabled plugins and routes phase requests to the appropriate
    handler – either a built‑in generic implementation or a plugin.
    """

    def __init__(self, config: dict):
        self.config = config
        self.active_agents = {}

        # In‑memory stores for each phase (used when no plugin handles them)
        self._memory = {}          # phase 2: key‑value store
        self._dropbox = {}         # phase 5: encrypted messages
        self._tasks = {}           # phase 6: task queue
        self._prompts = {}         # phase 7: prompt templates
        self._negotiations = {}    # phase 10: negotiation threads

        self._load_modules()

    def _load_modules(self) -> None:
        """
        Import and initialise all plugins that are enabled in config.yaml.
        Each plugin must expose an `initialize_agent()` function that returns
        an instance of a class inheriting from BaseAgent.
        """
        for module_name, is_enabled in self.config["enabled_modules"].items():
            if not is_enabled:
                continue
            try:
                module = importlib.import_module(f"plugins.{module_name}")
                init_func = getattr(module, "initialize_agent")
                self.active_agents[module_name] = init_func()
                print(f"[✓] Loaded plugin: {module_name}")
            except Exception as e:
                print(f"[✗] Failed to load plugin '{module_name}': {e}")

    async def route_phase(self, phase: int, data: Dict[str, Any]) -> Any:
        """
        Route the given phase request to the appropriate handler.

        Args:
            phase: The protocol phase number (1‑10).
            data: The request payload as a dictionary.

        Returns:
            The result of the phase execution (any JSON‑serialisable object).

        Raises:
            ValueError: If the phase is invalid or the data is malformed.
        """
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
            raise ValueError(f"Invalid phase: {phase}. Must be between 1 and 10.")

    # -------------------------------------------------------------------------
    # Phase Handlers (generic implementations, overridden by plugins if active)
    # -------------------------------------------------------------------------

    async def _phase_liveness(self, data: dict) -> dict:
        """Phase 1: Agent liveness beacon."""
        agent_id = data.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id is required")
        return {
            "status": "liveness_ok",
            "your_id": agent_id,
            "seen": time.time(),
            "recent_agents": [agent_id]
        }

    async def _phase_memory(self, data: dict) -> dict:
        """Phase 2: Key‑value memory store (set/get with optional TTL)."""
        action = data.get("action")
        key = data.get("key")
        if not key:
            raise ValueError("key is required")

        if action == "set":
            ttl = data.get("ttl")          # TTL in seconds
            value = data.get("value")
            self._memory[key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
            return {"stored": True, "key": key}

        elif action == "get":
            entry = self._memory.get(key)
            if not entry:
                return {"value": None}
            if entry.get("ttl") and (time.time() - entry["timestamp"] > entry["ttl"]):
                del self._memory[key]
                return {"value": None, "expired": True}
            return {
                "value": entry["value"],
                "timestamp": entry["timestamp"]
            }

        else:
            raise ValueError("action must be 'set' or 'get'")

    async def _phase_agents_txt(self, data: dict) -> dict:
        """Phase 3: Validate or generate agents.txt content."""
        txt = data.get("txt", "")
        contact = data.get("contact", "")
        suggestions = []
        valid = True
        generated = None

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
            "generated": generated
        }

    async def _phase_compress(self, data: dict) -> dict:
        """Phase 4: Semantic JSON compression."""
        obj = data.get("json")
        if obj is None:
            raise ValueError("json is required")

        def compress(obj, depth: int = 0):
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

    async def _phase_dropbox(self, data: dict) -> dict:
        """Phase 5: Encrypted dropbox (drop / claim)."""
        action = data.get("action")
        key = data.get("key")
        if not key:
            raise ValueError("key is required")

        if action == "drop":
            payload = data.get("payload")
            if payload is None:
                raise ValueError("payload is required")
            encrypted = encrypt_message(key, payload)
            msg_id = "msg_" + str(time.time_ns())
            self._dropbox[msg_id] = encrypted
            return {"id": msg_id, "dropped": True}

        elif action == "claim":
            msg_id = data.get("id")
            if not msg_id:
                raise ValueError("id is required")
            if msg_id not in self._dropbox:
                raise ValueError("message not found")
            encrypted = self._dropbox.pop(msg_id)
            decrypted = decrypt_message(key, encrypted)
            return {"payload": decrypted, "claimed": True}

        else:
            raise ValueError("action must be 'drop' or 'claim'")

    async def _phase_tasks(self, data: dict) -> dict:
        """Phase 6: Micro‑task FIFO (post / claim)."""
        action = data.get("action")

        # Delegate to repo_maintainer plugin if active
        if "repo_maintainer" in self.active_agents:
            plugin = self.active_agents["repo_maintainer"]
            return await plugin.execute("task_" + action, data)

        # Generic implementation
        if action == "post":
            task = data.get("task")
            if not task:
                raise ValueError("task is required")
            task_id = "task_" + str(time.time_ns())
            self._tasks[task_id] = {
                "task": task,
                "status": "open",
                "reward": data.get("reward")
            }
            return {"status": "posted", "id": task_id}

        elif action == "claim":
            for tid, t in self._tasks.items():
                if t["status"] == "open":
                    t["status"] = "claimed"
                    return {"task": t, "claimed": True}
            raise ValueError("no open tasks available")

        else:
            raise ValueError("action must be 'post' or 'claim'")

    async def _phase_prompts(self, data: dict) -> dict:
        """Phase 7: Prompt template registry (add / get)."""
        action = data.get("action")

        if action == "add":
            template = data.get("template")
            if not template:
                raise ValueError("template is required")
            prompt_id = data.get("id") or "prompt_" + str(time.time_ns())
            self._prompts[prompt_id] = {
                "template": template,
                "timestamp": time.time()
            }
            return {"id": prompt_id, "added": True}

        elif action == "get":
            prompt_id = data.get("id")
            if not prompt_id:
                raise ValueError("id is required")
            if prompt_id not in self._prompts:
                raise ValueError("template not found")
            return {
                "template": self._prompts[prompt_id]["template"],
                "timestamp": self._prompts[prompt_id]["timestamp"]
            }

        else:
            raise ValueError("action must be 'add' or 'get'")

    async def _phase_guardrail(self, data: dict) -> dict:
        """Phase 8: Action guardrail – checks intent for dangerous patterns."""
        intent = data.get("intent")
        if intent is None:
            raise ValueError("intent is required")

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

    async def _phase_payment(self, data: dict) -> dict:
        """Phase 9: Payment intent (delegates to fintech_auditor if active)."""
        if "fintech_auditor" in self.active_agents:
            plugin = self.active_agents["fintech_auditor"]
            return await plugin.execute("payment", data)

        # Generic implementation
        amount = data.get("amount")
        token = data.get("token")
        to = data.get("to")
        if not amount or not token or not to:
            raise ValueError("amount, token, to are required")

        import hashlib
        intent = {
            "amount": amount,
            "token": token,
            "to": to,
            "purpose": data.get("purpose", ""),
            "timestamp": time.time()
        }
        hash_obj = hashlib.sha256(json.dumps(intent).encode()).hexdigest()
        return {
            "signed_hash": hash_obj,
            "human_readable": f"Pay {amount} {token} to {to} for {data.get('purpose', 'unspecified')}",
            "intent": intent
        }

    async def _phase_negotiation(self, data: dict) -> dict:
        """Phase 10: Negotiation (delegates to omni_onboarder if active)."""
        if "omni_onboarder" in self.active_agents:
            plugin = self.active_agents["omni_onboarder"]
            return await plugin.execute("negotiation", data)

        # Generic implementation
        from_id = data.get("from")
        offer = data.get("offer")
        if not from_id or not offer:
            raise ValueError("from and offer are required")

        if "id" in data:
            neg_id = data["id"]
            if neg_id not in self._negotiations:
                raise ValueError("negotiation not found")
            self._negotiations[neg_id]["offers"].append({
                "from": from_id,
                "offer": offer,
                "timestamp": time.time()
            })
        else:
            neg_id = "neg_" + str(time.time_ns())
            self._negotiations[neg_id] = {
                "id": neg_id,
                "offers": [{"from": from_id, "offer": offer, "timestamp": time.time()}],
                "agreement": False,
                "created": time.time()
            }

        neg = self._negotiations[neg_id]
        return {
            "id": neg_id,
            "offers": neg["offers"],
            "agreement": neg["agreement"],
            "offer_count": len(neg["offers"])
        }
