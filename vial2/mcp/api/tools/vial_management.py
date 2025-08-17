import torch
import json
import uuid
from config.config import DatabaseConfig
from tools.agent_templates import AGENT_TEMPLATES
from postgrest import AsyncPostgrestClient
import logging

logger = logging.getLogger(__name__)

class VialManager:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.data_api = AsyncPostgrestClient("https://app-billowing-king-08029676.dpl.myneon.app")
        self.project_id = "twilight-art-21036984"

    async def execute(self, args: dict) -> dict:
        method = args.get("method")
        user_id = args.get("user_id")
        vial_id = args.get("vialId")
        project_id = args.get("project_id", self.project_id)
        if project_id != self.project_id:
            raise ValueError("Invalid Neon project ID")
        if method == "createVial":
            return await self.create_vial(user_id, vial_id, project_id)
        elif method == "prompt":
            return await self.process_prompt(user_id, vial_id, args.get("args"), project_id)
        elif method == "task":
            return await self.assign_task(user_id, vial_id, args.get("args"), project_id)
        elif method == "config":
            return await self.set_config(user_id, vial_id, args.get("args"), project_id)
        else:
            raise ValueError("Unknown vial method")

    async def execute_git(self, args: dict) -> dict:
        user_id = args.get("user_id")
        command = " ".join(args.get("args", []))
        project_id = args.get("project_id", self.project_id)
        try:
            result = {"status": "success", "output": f"Git command executed: {command}"}
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "git", json.dumps({"command": command}), str(uuid.uuid4()), project_id]
            )
            await self.data_api.from_("blocks").insert({"user_id": user_id, "type": "git", "data": json.dumps({"command": command}), "project_id": project_id}).eq("user_id", user_id).execute()
            logger.info(f"Git command executed for user {user_id}: {command}")
            return result
        except Exception as e:
            logger.error(f"Git command failed: {str(e)}")
            raise ValueError(f"Git command failed: {str(e)}")

    async def create_vial(self, user_id: str, vial_id: str, project_id: str) -> dict:
        template = AGENT_TEMPLATES.get(vial_id, AGENT_TEMPLATES["vial1"])
        await self.db.query(
            "INSERT INTO vials (vial_id, user_id, status, code, tasks, config, wallet_id, project_id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            [vial_id, user_id, "stopped", template, json.dumps([]), json.dumps({}), str(uuid.uuid4()), project_id]
        )
        await self.data_api.from_("vials").insert({"vial_id": vial_id, "user_id": user_id, "status": "stopped", "project_id": project_id}).eq("user_id", user_id).execute()
        logger.info(f"Vial {vial_id} created for user {user_id}")
        return {"status": "success", "vial_id": vial_id}

    async def process_prompt(self, user_id: str, vial_id: str, args: list, project_id: str) -> dict:
        prompt = " ".join(args)
        model = eval(AGENT_TEMPLATES[vial_id])()
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        await self.db.query(
            "UPDATE vials SET tasks = tasks || $1 WHERE vial_id = $2 AND user_id = $3 AND project_id = $4",
            [json.dumps([{"prompt": prompt, "output": output.tolist()}]), vial_id, user_id, project_id]
        )
        await self.data_api.from_("vials").update({"tasks": json.dumps([{"prompt": prompt, "output": output.tolist()}])}).eq("vial_id", vial_id).eq("user_id", user_id).eq("project_id", project_id).execute()
        logger.info(f"Prompt processed for vial {vial_id}")
        return {"status": "success", "output": output.tolist()}

    async def assign_task(self, user_id: str, vial_id: str, args: list, project_id: str) -> dict:
        task = " ".join(args)
        await self.db.query(
            "UPDATE vials SET tasks = tasks || $1 WHERE vial_id = $2 AND user_id = $3 AND project_id = $4",
            [json.dumps([{"task": task}]), vial_id, user_id, project_id]
        )
        await self.data_api.from_("vials").update({"tasks": json.dumps([{"task": task}])}).eq("vial_id", vial_id).eq("user_id", user_id).eq("project_id", project_id).execute()
        logger.info(f"Task assigned to vial {vial_id}")
        return {"status": "success", "task": task}

    async def set_config(self, user_id: str, vial_id: str, args: list, project_id: str) -> dict:
        key, value = args[:2]
        await self.db.query(
            "UPDATE vials SET config = config || $1 WHERE vial_id = $2 AND user_id = $3 AND project_id = $4",
            [json.dumps({key: value}), vial_id, user_id, project_id]
        )
        await self.data_api.from_("vials").update({"config": json.dumps({key: value})}).eq("vial_id", vial_id).eq("user_id", user_id).eq("project_id", project_id).execute()
        logger.info(f"Config updated for vial {vial_id}")
        return {"status": "success", "config": {key: value}}
