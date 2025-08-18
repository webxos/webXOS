import torch
from fastapi import HTTPException
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def handle_command(command: str, request: dict, db):
    try:
        parts = command.split()
        cmd = parts[0].lower()
        if cmd == '/prompt':
            vial_id, action = parts[1], ' '.join(parts[2:])
            async with db:
                await db.execute(
                    "UPDATE vials SET status=$1, code=$2, code_length=$3 WHERE vial_id=$4",
                    "running" if action.lower().find("train") != -1 else "stopped",
                    action,
                    len(action),
                    vial_id
                )
            return {"status": "success", "vial_id": vial_id, "action": action}
        elif cmd == '/task':
            vial_id, task = parts[1], ' '.join(parts[2:])
            async with db:
                await db.execute(
                    "UPDATE vials SET tasks=$1 WHERE vial_id=$2",
                    [task], vial_id
                )
            return {"status": "success", "vial_id": vial_id, "task": task}
        elif cmd == '/config':
            vial_id, key, value = parts[1], parts[2], ' '.join(parts[3:])
            async with db:
                await db.execute(
                    "UPDATE vials SET config=$1 WHERE vial_id=$2",
                    {key: value}, vial_id
                )
            return {"status": "success", "vial_id": vial_id, "config": {key: value}}
        elif cmd == '/status':
            async with db:
                vials = await db.fetch("SELECT * FROM vials")
                computes = await db.fetch("SELECT * FROM computes")
            return {"vials": [dict(v) for v in vials], "computes": [dict(c) for c in computes]}
        else:
            raise ValueError("Unknown command")
    except Exception as e:
        logger.error(f"Command handling failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def configure_compute(spec: dict, db):
    try:
        async with db:
            await db.execute(
                "UPDATE computes SET state=$1, spec=$2, last_activity=$3 WHERE compute_id=$4",
                "Configuration", spec, datetime.utcnow(), "compute1"
            )
        return {"status": "success", "spec": spec}
    except Exception as e:
        logger.error(f"Compute configuration failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def refresh_configuration(db):
    try:
        async with db:
            await db.execute(
                "UPDATE computes SET state=$1, last_activity=$2 WHERE compute_id=$3",
                "RefreshConfiguration", datetime.utcnow(), "compute1"
            )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Refresh configuration failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def terminate_fast(db):
    try:
        async with db:
            await db.execute(
                "UPDATE computes SET state=$1, readiness=$2, last_activity=$3 WHERE compute_id=$4",
                "TerminationPendingFast", False, datetime.utcnow(), "compute1"
            )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Fast termination failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def terminate_immediate(db):
    try:
        async with db:
            await db.execute(
                "UPDATE computes SET state=$1, readiness=$2, last_activity=$3 WHERE compute_id=$4",
                "TerminationPendingImmediate", False, datetime.utcnow(), "compute1"
            )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Immediate termination failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #agents #pytorch #neon_mcp
