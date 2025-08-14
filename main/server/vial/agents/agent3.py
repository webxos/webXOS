import logging,sqlite3,os
from datetime import datetime

logger=logging.getLogger(__name__)

class LLMwareAgent:
    """LLMwareAgent handles note management with wallet verification."""
    def __init__(self):
        """Initialize LLMwareAgent with notes directory."""
        self.notes_dir="/app/notes"
        os.makedirs(self.notes_dir,exist_ok=True)
        logger.info("LLMwareAgent initialized")

    async def add_note(self,wallet_id:str,content:str,resource_id:str|None=None)->dict:
        """Add a note to SQLite and file system.

        Args:
            wallet_id (str): Wallet ID for verification.
            content (str): Note content.
            resource_id (str, optional): Associated resource ID.

        Returns:
            dict: Success message with note ID.

        Raises:
            Exception: If note storage fails.
        """
        try:
            note_id=None
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT INTO notes (wallet_id,content,resource_id,timestamp) VALUES (?,?,?,?)",
                              (wallet_id,content,resource_id,datetime.now().isoformat()))
                note_id=cursor.lastrowid
                conn.commit()
            note_path=os.path.join(self.notes_dir,f"note_{note_id}_{wallet_id}.txt")
            with open(note_path,"w") as f:
                f.write(content)
            logger.info(f"Note {note_id} added for wallet {wallet_id}")
            return {"status":"success","note_id":note_id,"wallet_id":wallet_id}
        except Exception as e:
            logger.error(f"Note storage failed for wallet {wallet_id}: {str(e)}")
            raise Exception(f"Note storage failed: {str(e)}")

    async def read_note(self,note_id:int,wallet_id:str)->dict:
        """Read a note from SQLite by ID and wallet.

        Args:
            note_id (int): ID of the note to read.
            wallet_id (str): Wallet ID for verification.

        Returns:
            dict: Note content and metadata.

        Raises:
            Exception: If note retrieval fails or unauthorized.
        """
        try:
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE id=? AND wallet_id=?",(note_id,wallet_id))
                note=cursor.fetchone()
                if not note:
                    logger.warning(f"Note {note_id} not found or unauthorized for wallet {wallet_id}")
                    raise Exception("Note not found or unauthorized")
                logger.info(f"Note {note_id} read for wallet {wallet_id}")
                return {"status":"success","note":{"id":note[0],"content":note[1],"resource_id":note[2],"timestamp":note[3],"wallet_id":wallet_id}}
        except Exception as e:
            logger.error(f"Note read failed for wallet {wallet_id}: {str(e)}")
            raise Exception(f"Note read failed: {str(e)}")