import logging,sqlite3,os
from fastapi import HTTPException,Depends
from pydantic import BaseModel
from .mcp_auth_server import MCPAuthServer
from datetime import datetime

logger=logging.getLogger(__name__)

class NoteRequest(BaseModel):
    wallet_id:str
    content:str
    resource_id:str|None=None

class NoteReadRequest(BaseModel):
    note_id:int
    wallet_id:str

class MCPNotesHandler:
    """Manages note creation and retrieval with wallet verification."""
    def __init__(self):
        """Initialize MCPNotesHandler with notes directory."""
        self.notes_dir="/app/notes"
        self.auth_server=MCPAuthServer()
        os.makedirs(self.notes_dir,exist_ok=True)
        logger.info("MCPNotesHandler initialized")

    async def add_note(self,request:NoteRequest,access_token:str=Depends(lambda x: x)) -> dict:
        """Add a note to SQLite and file system.

        Args:
            request (NoteRequest): Note data with wallet_id, content, and optional resource_id.
            access_token (str): OAuth access token.

        Returns:
            dict: Success message with note ID.

        Raises:
            HTTPException: If note storage or token verification fails.
        """
        try:
            if not await self.auth_server.verify_oauth_token(access_token,request.wallet_id):
                logger.warning(f"Invalid token for wallet {request.wallet_id}")
                raise HTTPException(status_code=401,detail="Invalid access token")
            note_id=None
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT INTO notes (wallet_id,content,resource_id,timestamp) VALUES (?,?,?,?)",
                              (request.wallet_id,request.content,request.resource_id,datetime.now().isoformat()))
                note_id=cursor.lastrowid
                conn.commit()
            note_path=os.path.join(self.notes_dir,f"note_{note_id}_{request.wallet_id}.txt")
            with open(note_path,"w") as f:
                f.write(request.content)
            logger.info(f"Note {note_id} added for wallet {request.wallet_id}")
            return {"status":"success","note_id":note_id,"wallet_id":request.wallet_id}
        except Exception as e:
            logger.error(f"Note storage failed for wallet {request.wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Note storage failed: {str(e)}")

    async def read_note(self,request:NoteReadRequest,access_token:str=Depends(lambda x: x)) -> dict:
        """Read a note from SQLite by ID and wallet.

        Args:
            request (NoteReadRequest): Request with note_id and wallet_id.
            access_token (str): OAuth access token.

        Returns:
            dict: Note content and metadata.

        Raises:
            HTTPException: If note retrieval or token verification fails.
        """
        try:
            if not await self.auth_server.verify_oauth_token(access_token,request.wallet_id):
                logger.warning(f"Invalid token for wallet {request.wallet_id}")
                raise HTTPException(status_code=401,detail="Invalid access token")
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE id=? AND wallet_id=?",(request.note_id,request.wallet_id))
                note=cursor.fetchone()
                if not note:
                    logger.warning(f"Note {request.note_id} not found or unauthorized for wallet {request.wallet_id}")
                    raise HTTPException(status_code=404,detail="Note not found or unauthorized")
                logger.info(f"Read note {request.note_id} for wallet {request.wallet_id}")
                return {"status":"success","note":{"id":note[0],"content":note[1],"resource_id":note[2],"timestamp":note[3],"wallet_id":request.wallet_id}}
        except Exception as e:
            logger.error(f"Note read failed for wallet {request.wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Note read failed: {str(e)}")
