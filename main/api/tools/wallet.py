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
@@ -78,6 +85,9 @@ async def execute(self, input: Dict[str, Any]) -> Any:
elif method == "importWallet":
import_input = WalletImportInput(**input)
return await self.import_wallet(import_input)
            elif method == "batchSync":
                sync_input = WalletBatchSyncInput(**input)
                return await self.batch_sync(sync_input)
elif method == "exportVials":
export_input = WalletBalanceInput(**input)
return await self.export_vials(export_input)
@@ -113,7 +123,6 @@ async def get_vial_balance(self, input: WalletBalanceInput) -> WalletBalanceOutp

async def import_wallet(self, input: WalletImportInput) -> WalletImportOutput:
try:
            # Validate markdown hash
calculated_hash = hashlib.sha256(input.markdown.encode()).hexdigest()
if calculated_hash != input.hash:
raise ValidationError("Invalid markdown file: Hash mismatch")
@@ -145,6 +154,54 @@ async def import_wallet(self, input: WalletImportInput) -> WalletImportOutput:
logger.error(f"Import wallet error: {str(e)}")
raise HTTPException(400, str(e))

    async def batch_sync(self, input: WalletBatchSyncInput) -> WalletBatchSyncOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            current_balance = float(user.rows[0]["balance"])
            results = []
            
            for op in input.operations:
                if op["method"] == "importWallet":
                    calculated_hash = hashlib.sha256(op["markdown"].encode()).hexdigest()
                    if calculated_hash != op["hash"]:
                        results.append({"error": "Invalid markdown file: Hash mismatch"})
                        continue
                    
                    balances = []
                    for line in op["markdown"].splitlines():
                        if match := re.match(r".*balance\s*=\s*(\d+\.\d+)", line):
                            balances.append(float(match.group(1)))
                    
                    total_balance = sum(balances)
                    current_balance += total_balance
                    results.append({"imported_vials": [f"vial{i+1}" for i in range(len(balances))], "total_balance": current_balance})
                
                elif op["method"] == "mineVial":
                    data = f"{input.user_id}{op['vial_id']}{op['nonce']}"
                    hash_value = hashlib.sha256(data.encode()).hexdigest()
                    difficulty = 2
                    reward = 0.0
                    
                    if hash_value.startswith("0" * difficulty):
                        reward = 1.0
                        current_balance += reward
                    
                    results.append({"hash": hash_value, "reward": reward, "balance": current_balance})
            
            await self.db.query(
                "UPDATE users SET balance = $1 WHERE user_id = $2",
                [current_balance, input.user_id]
            )
            
            logger.info(f"Batch synced operations for {input.user_id}, new balance: {current_balance}")
            return WalletBatchSyncOutput(results=results)
        except Exception as e:
            logger.error(f"Batch sync error: {str(e)}")
            raise HTTPException(400, str(e))

async def export_vials(self, input: WalletBalanceInput) -> WalletExportOutput:
try:
user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
