import json,logging,os,random

logger=logging.getLogger(__name__)

class PromptManager:
    """Manages automated prompt selection from a prompt pool."""
    def __init__(self,prompt_file="/app/main/server/prompts/prompt_pool.json"):
        """Initialize PromptManager with prompt pool file.

        Args:
            prompt_file (str): Path to prompt pool JSON file.
        """
        self.prompt_file=prompt_file
        self.prompts=self.load_prompts()
        logger.info("PromptManager initialized")

    def load_prompts(self)->list:
        """Load prompts from JSON file.

        Returns:
            list: List of prompt dictionaries.

        Raises:
            Exception: If prompt file loading fails.
        """
        try:
            if not os.path.exists(self.prompt_file):
                logger.warning("Prompt pool file not found, returning empty list")
                return []
            with open(self.prompt_file,"r") as f:
                data=json.load(f)
                logger.info(f"Loaded {len(data['prompts'])} prompts from {self.prompt_file}")
                return data["prompts"]
        except Exception as e:
            logger.error(f"Failed to load prompts: {str(e)}")
            raise Exception(f"Failed to load prompts: {str(e)}")

    def get_prompt_for_agent(self,agent_name:str,wallet_id:str,vial_id:str|None=None,content:str|None=None)->str:
        """Get a prompt for a specific agent, substituting parameters.

        Args:
            agent_name (str): Name of the agent (e.g., NomicAgent).
            wallet_id (str): Wallet ID for substitution.
            vial_id (str, optional): Vial ID for substitution.
            content (str, optional): Content for substitution.

        Returns:
            str: Formatted prompt string.

        Raises:
            Exception: If no suitable prompt is found.
        """
        try:
            agent_prompts=[p for p in self.prompts if p["agent"]==agent_name]
            if not agent_prompts:
                logger.warning(f"No prompts found for agent {agent_name}")
                raise Exception(f"No prompts found for agent {agent_name}")
            prompt=random.choice(agent_prompts)["content"]
            prompt=prompt.replace("{wallet_id}",wallet_id)
            if vial_id:
                prompt=prompt.replace("{vial_id}",vial_id)
            if content:
                prompt=prompt.replace("{content}",content)
            logger.info(f"Generated prompt for {agent_name}: {prompt}")
            return prompt
        except Exception as e:
            logger.error(f"Prompt generation failed for {agent_name}: {str(e)}")
            raise Exception(f"Prompt generation failed: {str(e)}")
