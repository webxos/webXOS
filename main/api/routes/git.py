from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ...config.mcp_config import mcp_config
import git
import os
import shutil

router = APIRouter()

class GitRequest(BaseModel):
    repo_url: str
    branch: str = "main"

class GitResponse(BaseModel):
    status: str
    output: str

@router.post("/git/push")
async def git_push(request: GitRequest, user_id: str = Depends(verify_token)):
    try:
        repo_dir = f"/tmp/repo_{user_id}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        repo = git.Repo.clone_from(mcp_config.REPO_URL, repo_dir, branch=request.branch)
        repo.git.add(all=True)
        repo.git.commit(m="Automated commit from WEBXOS MCP Gateway")
        repo.git.push("origin", request.branch, env={"GIT_ASKPASS": mcp_config.REPO_TOKEN})
        log_info(f"Git push successful for {user_id} on {request.branch}")
        return GitResponse(status="success", output="Pushed to repository")
    except Exception as e:
        log_error(f"Git push failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Git error: {str(e)}")

@router.post("/git/pull")
async def git_pull(request: GitRequest, user_id: str = Depends(verify_token)):
    try:
        repo_dir = f"/tmp/repo_{user_id}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        repo = git.Repo.clone_from(mcp_config.REPO_URL, repo_dir, branch=request.branch)
        repo.git.pull("origin", request.branch, env={"GIT_ASKPASS": mcp_config.REPO_TOKEN})
        log_info(f"Git pull successful for {user_id} on {request.branch}")
        return GitResponse(status="success", output="Pulled from repository")
    except Exception as e:
        log_error(f"Git pull failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Git error: {str(e)}")

@router.post("/git/diff")
async def git_diff(request: GitRequest, user_id: str = Depends(verify_token)):
    try:
        repo_dir = f"/tmp/repo_{user_id}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        repo = git.Repo.clone_from(mcp_config.REPO_URL, repo_dir, branch=request.branch)
        diff = repo.git.diff()
        log_info(f"Git diff retrieved for {user_id} on {request.branch}")
        return GitResponse(status="success", output=diff)
    except Exception as e:
        log_error(f"Git diff failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Git error: {str(e)}")

@router.post("/git/commit")
async def git_commit(request: GitRequest, user_id: str = Depends(verify_token)):
    try:
        repo_dir = f"/tmp/repo_{user_id}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        repo = git.Repo.clone_from(mcp_config.REPO_URL, repo_dir, branch=request.branch)
        repo.git.add(all=True)
        repo.git.commit(m="Automated commit from WEBXOS MCP Gateway")
        log_info(f"Git commit successful for {user_id} on {request.branch}")
        return GitResponse(status="success", output="Committed changes")
    except Exception as e:
        log_error(f"Git commit failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Git error: {str(e)}")

@router.post("/git/log")
async def git_log(request: GitRequest, user_id: str = Depends(verify_token)):
    try:
        repo_dir = f"/tmp/repo_{user_id}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        repo = git.Repo.clone_from(mcp_config.REPO_URL, repo_dir, branch=request.branch)
        log = repo.git.log(max_count=10)
        log_info(f"Git log retrieved for {user_id} on {request.branch}")
        return GitResponse(status="success", output=log)
    except Exception as e:
        log_error(f"Git log failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Git error: {str(e)}")

@router.post("/git/checkout")
async def git_checkout(request: GitRequest, user_id: str = Depends(verify_token)):
    try:
        repo_dir = f"/tmp/repo_{user_id}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        repo = git.Repo.clone_from(mcp_config.REPO_URL, repo_dir, branch=request.branch)
        repo.git.checkout(request.branch)
        log_info(f"Git checkout successful for {user_id} on {request.branch}")
        return GitResponse(status="success", output=f"Checked out {request.branch}")
    except Exception as e:
        log_error(f"Git checkout failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Git error: {str(e)}")
