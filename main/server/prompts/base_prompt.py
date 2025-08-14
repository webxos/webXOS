import json,logging,os

logger=logging.getLogger(__name__)

class BasePromptManager:
    """Manages prompt automation for Vial MCP agents."""
    def __init__(self,prompt_file="/app/main/server/prompts/prompt_pool.json"):
        """Initialize BasePromptManager with prompt file path.

        Args:
            prompt_file (str): Path to prompt pool JSON file.
        """
        self.prompt_file=prompt_file
        self.prompts=self.load_prompts()
        logger.info("BasePromptManager initialized")

    def load_prompts(self):
        """Load prompts from prompt_pool.json.

        Returns:
            list: List of prompt dictionaries.

        Raises:
            Exception: If prompt file loading fails.
        """
        try:
            if not os.path.exists(self.prompt_file):
                logger.warning(f"Prompt file {self.prompt_file} not found")
                return []
            with open(self.prompt_file,"r") as f:
                prompts=json.load(f).get("prompts",[])
            logger.info(f"Loaded {len(prompts)} prompts from {self.prompt_file}")
            return prompts
        except Exception as e:
            logger.error(f"Prompt loading failed: {str(e)}")
            with open("/app/errorlog.md","a") as f:
                f.write(f"[{datetime.now().isoformat()}] [BasePromptManager] Prompt loading failed: {str(e)}\n")
            raise Exception(f"Prompt loading failed: {str(e)}")

    def get_prompt(self,agent_name,wallet_id=None,vial_id=None,content=None):
        """Generate a prompt for a specific agent with optional substitutions.

        Args:
            agent_name (str): Name of the agent (e.g., NomicAgent).
            wallet_id (str, optional): Wallet ID for substitution.
            vial_id (str, optional): Vial ID for substitution.
            content (str, optional): Content for substitution.

        Returns:
            str: Formatted prompt.

        Raises:
            Exception: If prompt generation fails.
        """
        try:
            for prompt in self.prompts:
                if prompt["agent"]==agent_name:
                    result=prompt["content"]
                    if wallet_id:
                        result=result.replace("{wallet_id}",wallet_id)
                    if vial_id:
                        result=result.replace("{vial_id}",vial_id)
                    if content:
                        result=result.replace("{content}",content)
                    logger.info(f"Generated prompt for {agent_name}")
                    return result
            logger.warning(f"No prompt found for agent {agent_name}")
            return ""
        except Exception as e:
            logger.error(f"Prompt generation failed for {agent_name}: {str(e)}")
            with open("/app/errorlog.md","a") as f:
                f.write(f"[{datetime.now().isoformat()}] [BasePromptManager] Prompt generation failed: {str(e)}\n")
            raise Exception(f"Prompt generation failed: {str(e)}")
