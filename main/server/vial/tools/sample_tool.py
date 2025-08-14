import logging,os
from datetime import datetime

logger=logging.getLogger(__name__)

class SampleTool:
    """Sample extensible tool for Vial MCP."""
    def __init__(self,output_dir="/app/notes"):
        """Initialize SampleTool with output directory.

        Args:
            output_dir (str): Directory for tool output.
        """
        self.output_dir=output_dir
        os.makedirs(self.output_dir,exist_ok=True)
        logger.info("SampleTool initialized")

    def generate_sample_note(self,wallet_id:str,content:str) -> dict:
        """Generate a sample note and save it to the notes directory.

        Args:
            wallet_id (str): Wallet ID for the note.
            content (str): Content of the note.

        Returns:
            dict: Success message with note path.

        Raises:
            Exception: If note generation fails.
        """
        try:
            note_id=f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            note_path=os.path.join(self.output_dir,f"{note_id}.txt")
            with open(note_path,"w") as f:
                f.write(f"Wallet: {wallet_id}\nContent: {content}\nTimestamp: {datetime.now().isoformat()}")
            logger.info(f"Generated sample note {note_id} for wallet {wallet_id}")
            return {"status":"success","note_id":note_id,"note_path":note_path}
        except Exception as e:
            logger.error(f"Sample note generation failed: {str(e)}")
            with open("/app/errorlog.md","a") as f:
                f.write(f"[{datetime.now().isoformat()}] [SampleTool] Sample note generation failed: {str(e)}\n")
            raise Exception(f"Sample note generation failed: {str(e)}")
