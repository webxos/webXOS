import logging
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from vial.webxos_wallet import WebXOSWallet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainAgent:
    def __init__(self):
        self.wallet = WebXOSWallet()
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Enhance this query for better AI processing: {query}"
        )
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=None)  # Placeholder for actual LLM

    def enhance_query(self, query: str) -> str:
        try:
            enhanced = self.llm_chain.run(query=query) if self.llm_chain.llm else query + " (enhanced)"
            logger.info(f"Enhanced query: {query} -> {enhanced}")
            return enhanced
        except Exception as e:
            logger.error(f"Query enhancement error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Query enhancement error: {str(e)}\n")
            return query

    def train_vial(self, vial_id: str, query: str, wallet_balance: float) -> bool:
        try:
            if wallet_balance < 0.0001:
                raise ValueError("Insufficient $WEBXOS for training")
            enhanced_query = self.enhance_query(query)
            self.wallet.update_balance(vial_id, wallet_balance - 0.0001)
            logger.info(f"Trained vial {vial_id} with query: {enhanced_query}")
            return True
        except Exception as e:
            logger.error(f"Vial training error for {vial_id}: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Vial training error: {str(e)}\n")
            return False
