from config.config import DatabaseConfig
from lib.errors import ValidationError
from tools.agent_templates import get_all_agents
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any, List
import hashlib
import re
import uuid
from datetime import datetime
import json

logger = logging.getLogger("mcp.wallet")
logger.setLevel(logging.INFO)

class WalletBalanceInput(BaseModel):
    user_id: str
    vial_id: str

class WalletBalanceOutput(BaseModel):
    vial_id: str
    balance: float

class WalletImportInput(BaseModel):
    user_id: str
    markdown: str
    hash: str

class WalletImportOutput(BaseModel):
    imported_vials: List[str]
    total_balance: float

class WalletBatchSyncInput(BaseModel):
    user_id: str
    operations: List[Dict[str, Any]]

class WalletBatchSyncOutput(BaseModel):
    results: List[Dict[str, Any]]

class WalletExportOutput(BaseModel):
    markdown: str
    hash: str

class WalletMineInput(BaseModel):
    user_id: str
    vial_id: str
    nonce: int

class WalletMineOutput(BaseModel):
    hash: str
    reward: float
    balance: float

class WalletVoidInput(BaseModel):
    user_id: str
    vial_id: str

class WalletVoidOutput(BaseModel):
    vial_id: str
    status: str

class WalletTroubleshootInput(BaseModel):
    user_id: str
    vial_id: str

class WalletTroubleshootOutput(BaseModel):
    vial_id: str
    status: str
    diagnostics: Dict[str, Any]

class WalletQuantumLinkInput(BaseModel):
    user_id: str

class WalletQuantumLinkOutput(BaseModel):
    link_id: str

class WalletCashOutInput(BaseModel):
    user_id: str
    amount: float
    destination_address: str

class WalletCashOutOutput(BaseModel):
    transaction_id: str
    new_balance: float

class WalletTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.agents = {agent.get_metadata()["vial_id"]: agent.get_metadata() for agent in get_all_agents()}

    async def initialize_new_wallet(self, user_id: str, wallet_address: str, api_key: str, api_secret: str):
        await self.db.query(
            "INSERT INTO users (user_id, balance, wallet_address, api_key, api_secret, reputation) VALUES ($1, $2, $3, $4, $5, $6)",
            [user_id, 0.0, wallet_address, api_key, api_secret, 0]
        )
        for agent in self.agents.values():
            await self.db.query(
                "INSERT INTO vials (user_id, vial_id, code, wallet_address, webxos_hash) VALUES ($1, $2, $3, $4, $5)",
                [user_id, agent["vial_id"], agent["code"], agent["wallet_address"], agent["webxos_hash"]]
            )

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            method = input.get("method", "getVialBalance")
            if method == "getVialBalance":
                wallet_input = WalletBalanceInput(**input)
                return await self.get_vial_balance(wallet_input)
            elif method == "importWallet":
                import_input = WalletImportInput(**input)
                return await self.import_wallet(import_input)
            elif method == "batchSync":
                sync_input = WalletBatchSyncInput(**input)
                return await self.batch_sync(sync_input)
            elif method == "exportVials":
                export_input = WalletBalanceInput(**input)
                return await self.export_vials(export_input)
            elif method == "mineVial":
                mine_input = WalletMineInput(**input)
                return await self.mine_vial(mine_input)
            elif method == "voidVial":
                void_input = WalletVoidInput(**input)
                return await self.void_vial(void_input)
            elif method == "troubleshootVial":
                troubleshoot_input = WalletTroubleshootInput(**input)
                return await self.troubleshoot_vial(troubleshoot_input)
            elif method == "quantumLink":
                quantum_input = WalletQuantumLinkInput(**input)
                return await self.quantum_link(quantum_input)
            elif method == "cashOut":
                cash_out_input = WalletCashOutInput(**input)
                return await self.cash_out(cash_out_input)
            else:
                raise ValidationError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Wallet error: {str(e)}")
            raise HTTPException(400, str(e))

    async def get_vial_balance(self, input: WalletBalanceInput) -> WalletBalanceOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            total_balance = float(user.rows[0]["balance"])
            vial_balance = total_balance / 4
            logger.info(f"Retrieved vial balance for {input.user_id}, vial {input.vial_id}: {vial_balance}")
            return WalletBalanceOutput(vial_id=input.vial_id, balance=vial_balance)
        except Exception as e:
            logger.error(f"Get vial balance error: {str(e)}")
            raise HTTPException(400, str(e))

    async def import_wallet(self, input: WalletImportInput) -> WalletImportOutput:
        try:
            calculated_hash = hashlib.sha256(input.markdown.encode()).hexdigest()
            if calculated_hash != input.hash:
                raise ValidationError("Invalid markdown file: Hash mismatch")
            
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            balances = []
            vials = []
            for section in input.markdown.split("---"):
                if match := re.search(r"Wallet Balance: (\d+\.\d{4}) \$WEBXOS", section):
                    balances.append(float(match.group(1)))
                if match := re.search(r"# Vial Agent: (vial\d+)", section):
                    vials.append(match.group(1))
                if match := re.search(r"```python\n([\s\S]+?)\n```", section):
                    code = match.group(1)
                    vial_id = section.split("Vial Agent: ")[1].split("\n")[0]
                    webxos_hash = hashlib.sha256(f"{vial_id}{uuid.uuid4()}".encode()).hexdigest()
                    await self.db.query(
                        "UPDATE vials SET code = $1, webxos_hash = $2 WHERE user_id = $3 AND vial_id = $4",
                        [code, webxos_hash, input.user_id, vial_id]
                    )
            
            total_balance = sum(balances)
            current_balance = float(user.rows[0]["balance"])
            new_balance = current_balance + total_balance
            
            await self.db.query(
                "UPDATE users SET balance = $1 WHERE user_id = $2",
                [new_balance, input.user_id]
            )
            
            logger.info(f"Imported wallet for {input.user_id}, new balance: {new_balance}")
            return WalletImportOutput(
                imported_vials=vials,
                total_balance=new_balance
            )
        except Exception as e:
            logger.error(f"Import wallet error: {str(e)}")
            raise HTTPException(400, str(e))

    async def batch_sync(self, input: WalletBatchSyncInput) -> WalletBatchSyncOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            current_balance = float(user.rows[0]["balance"])
            results = []
            import_vials = []
            mining_ops = []
            
            for op in input.operations:
                if op["method"] == "importWallet":
                    calculated_hash = hashlib.sha256(op["markdown"].encode()).hexdigest()
                    if calculated_hash != op["hash"]:
                        results.append({"error": "Invalid markdown file: Hash mismatch"})
                        continue
                    
                    balances = []
                    vials = []
                    for section in op["markdown"].split("---"):
                        if match := re.search(r"Wallet Balance: (\d+\.\d{4}) \$WEBXOS", section):
                            balances.append(float(match.group(1)))
                        if match := re.search(r"# Vial Agent: (vial\d+)", section):
                            vials.append(match.group(1))
                        if match := re.search(r"```python\n([\s\S]+?)\n```", section):
                            code = match.group(1)
                            vial_id = section.split("Vial Agent: ")[1].split("\n")[0]
                            webxos_hash = hashlib.sha256(f"{vial_id}{uuid.uuid4()}".encode()).hexdigest()
                            await self.db.query(
                                "UPDATE vials SET code = $1, webxos_hash = $2 WHERE user_id = $3 AND vial_id = $4",
                                [code, webxos_hash, input.user_id, vial_id]
                            )
                    
                    total_balance = sum(balances)
                    import_vials.append((total_balance, vials))
                
                elif op["method"] == "mineVial":
                    mining_ops.append((op["vial_id"], op["nonce"]))
            
            if import_vials:
                total_import_balance = sum(v[0] for v in import_vials)
                current_balance += total_import_balance
                await self.db.query(
                    "UPDATE users SET balance = $1 WHERE user_id = $2",
                    [current_balance, input.user_id]
                )
                for total_balance, vials in import_vials:
                    results.append({"imported_vials": vials, "total_balance": current_balance})
            
            for vial_id, nonce in mining_ops:
                data = f"{input.user_id}{vial_id}{nonce}"
                hash_value = hashlib.sha256(data.encode()).hexdigest()
                difficulty = 2
                reward = 0.0
                
                if hash_value.startswith("0" * difficulty):
                    reward = 1.0
                    current_balance += reward
                    await self.db.query(
                        "UPDATE users SET balance = $1 WHERE user_id = $2",
                        [current_balance, input.user_id]
                    )
                
                results.append({"hash": hash_value, "reward": reward, "balance": current_balance})
            
            logger.info(f"Batch synced operations for {input.user_id}, new balance: {current_balance}")
            return WalletBatchSyncOutput(results=results)
        except Exception as e:
            logger.error(f"Batch sync error: {str(e)}")
            raise HTTPException(400, str(e))

    async def export_vials(self, input: WalletBalanceInput) -> WalletExportOutput:
        try:
            user = await self.db.query("SELECT user_id, balance, wallet_address, api_key, api_secret, reputation FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            total_balance = float(user.rows[0]["balance"])
            wallet_address = user.rows[0]["wallet_address"]
            api_key = user.rows[0]["api_key"]
            api_secret = user.rows[0]["api_secret"]
            reputation = user.rows[0]["reputation"]
            network_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3]
            vial_balance = total_balance / 4
            blocks = 1958
            hash_value = hashlib.sha256(f"{input.user_id}{wallet_address}{timestamp}".encode()).hexdigest()
            
            vials = await self.db.query("SELECT vial_id, code, wallet_address, webxos_hash FROM vials WHERE user_id = $1", [input.user_id])
            vial_data = {row["vial_id"]: row for row in vials.rows} if vials.rows else self.agents
            
            markdown = f"""# WebXOS Vial and Wallet Export

## Agentic Network
- Network ID: {network_id}
- Session Start: {timestamp}
- Session Duration: 0.00 seconds
- Reputation: {reputation}

## Wallet
- Wallet Key: {str(uuid.uuid4())}
- Session Balance: {total_balance:.4f} $WEBXOS
- Address: {wallet_address}
- Hash: {hash_value}

## API Credentials
- Key: {api_key}
- Secret: {api_secret}

## Blockchain
- Blocks: {blocks}
- Last Hash: {hash_value}

## Vials
"""
            for vial_id in ["vial1", "vial2", "vial3", "vial4"]:
                agent = vial_data.get(vial_id, self.agents[vial_id])
                markdown += f"""# Vial Agent: {vial_id}
- Status: running
- Language: Python
- Code Length: {agent["code_length"]} bytes
- $WEBXOS Hash: {agent["webxos_hash"]}
- Wallet Balance: {vial_balance:.4f} $WEBXOS
- Wallet Address: {agent["wallet_address"]}
- Wallet Hash: {hash_value}
- Tasks: none
- Quantum State: {json.dumps(agent["quantum_state"], indent=6)}
- Training Data: {json.dumps(agent["training_data"], indent=6)}
- Config: {{}}

```python
{agent["code"]}
```

---
"""
            markdown += """## Instructions
- **Reuse**: Import this .md file via the "Import" button to resume training.
- **Extend**: Modify agent code externally, then reimport.
- **Share**: Send this .md file to others to continue training with the same wallet.
- **API**: Use API credentials with LangChain to train vials (online mode only).
- **Cash Out**: $WEBXOS balance and reputation are tied to the wallet address and hash for secure verification (online mode only).

Generated by Vial MCP Controller
"""
            final_hash = hashlib.sha256(markdown.encode()).hexdigest()
            logger.info(f"Exported vials for {input.user_id}")
            return WalletExportOutput(markdown=markdown, hash=final_hash)
        except Exception as e:
            logger.error(f"Export vials error: {str(e)}")
            raise HTTPException(400, str(e))

    async def mine_vial(self, input: WalletMineInput) -> WalletMineOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            data = f"{input.user_id}{input.vial_id}{input.nonce}"
            hash_value = hashlib.sha256(data.encode()).hexdigest()
            difficulty = 2
            reward = 0.0
            
            if hash_value.startswith("0" * difficulty):
                reward = 1.0
                current_balance = float(user.rows[0]["balance"])
                new_balance = current_balance + reward
                await self.db.query(
                    "UPDATE users SET balance = $1 WHERE user_id = $2",
                    [new_balance, input.user_id]
                )
                logger.info(f"Mining successful for {input.user_id}, vial {input.vial_id}, reward: {reward}")
            else:
                new_balance = float(user.rows[0]["balance"])
            
            return WalletMineOutput(hash=hash_value, reward=reward, balance=new_balance)
        except Exception as e:
            logger.error(f"Mine vial error: {str(e)}")
            raise HTTPException(400, str(e))

    async def void_vial(self, input: WalletVoidInput) -> WalletVoidOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            await self.db.query(
                "UPDATE users SET balance = 0 WHERE user_id = $1",
                [input.user_id]
            )
            
            logger.info(f"Voided vial {input.vial_id} for {input.user_id}")
            return WalletVoidOutput(vial_id=input.vial_id, status="voided")
        except Exception as e:
            logger.error(f"Void vial error: {str(e)}")
            raise HTTPException(400, str(e))

    async def troubleshoot_vial(self, input: WalletTroubleshootInput) -> WalletTroubleshootOutput:
        try:
            user = await self.db.query("SELECT user_id, balance, wallet_address FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            diagnostics = {
                "balance": float(user.rows[0]["balance"]),
                "wallet_address": user.rows[0]["wallet_address"],
                "active": user.rows[0]["balance"] > 0
            }
            
            logger.info(f"Troubleshooted vial {input.vial_id} for {input.user_id}")
            return WalletTroubleshootOutput(
                vial_id=input.vial_id,
                status="operational" if diagnostics["active"] else "inactive",
                diagnostics=diagnostics
            )
        except Exception as e:
            logger.error(f"Troubleshoot vial error: {str(e)}")
            raise HTTPException(400, str(e))

    async def quantum_link(self, input: WalletQuantumLinkInput) -> WalletQuantumLinkOutput:
        try:
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            link_id = str(uuid.uuid4())
            await self.db.query(
                "INSERT INTO quantum_links (link_id, user_id, quantum_state) VALUES ($1, $2, $3)",
                [link_id, input.user_id, json.dumps({"qubits": [], "entanglement": "synced"})]
            )
            logger.info(f"Established quantum link for {input.user_id}: {link_id}")
            return WalletQuantumLinkOutput(link_id=link_id)
        except Exception as e:
            logger.error(f"Quantum link error: {str(e)}")
            raise HTTPException(400, str(e))

    async def cash_out(self, input: WalletCashOutInput) -> WalletCashOutOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            current_balance = float(user.rows[0]["balance"])
            if input.amount <= 0 or input.amount > current_balance:
                raise ValidationError("Invalid cash-out amount")
            
            if not re.match(r'^[a-f0-9]{64}$', input.destination_address):
                raise ValidationError("Invalid destination address")
            
            new_balance = current_balance - input.amount
            transaction_id = str(uuid.uuid4())
            
            await self.db.query(
                "UPDATE users SET balance = $1 WHERE user_id = $2",
                [new_balance, input.user_id]
            )
            
            # Log transaction (simplified, extend with blockchain integration)
            await self.db.query(
                "INSERT INTO transactions (transaction_id, user_id, amount, destination_address, timestamp) VALUES ($1, $2, $3, $4, $5)",
                [transaction_id, input.user_id, input.amount, input.destination_address, datetime.utcnow()]
            )
            
            logger.info(f"Cashed out {input.amount} $WEBXOS for {input.user_id} to {input.destination_address}")
            return WalletCashOutOutput(transaction_id=transaction_id, new_balance=new_balance)
        except Exception as e:
            logger.error(f"Cash out error: {str(e)}")
            raise HTTPException(400, str(e))
