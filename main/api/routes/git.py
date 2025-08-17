from fastapi import APIRouter, Depends
from pydantic import BaseModel
from utils.auth import verify_token

router = APIRouter()

class GitRequest(BaseModel):
    repo_url: str
    branch: str

@router.post("/git/push")
async def git_push(request: GitRequest, token: str = Depends(verify_token)):
    return {"status": "success", "output": f"Pushed to {request.repo_url} ({request.branch})"}

@router.post("/git/pull")
async def git_pull(request: GitRequest, token: str = Depends(verify_token)):
    return {"status": "success", "output": f"Pulled from {request.repo_url} ({request.branch})"}

@router.post("/git/diff")
async def git_diff(request: GitRequest, token: str = Depends(verify_token)):
    return {"status": "success", "output": "Diff output"}

@router.post("/git/commit")
async def git_commit(request: GitRequest, token: str = Depends(verify_token)):
    return {"status": "success", "output": "Commit completed"}

@router.post("/git/log")
async def git_log(request: GitRequest, token: str = Depends(verify_token)):
    return {"status": "success", "output": "Commit log"}

@router.post("/git/checkout")
async def git_checkout(request: GitRequest, token: str = Depends(verify_token)):
    return {"status": "success", "output": f"Checked out {request.branch}"}
