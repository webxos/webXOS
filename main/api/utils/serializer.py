from main.api.utils.logging import logger

class MarkdownSerializer:
    @staticmethod
    def serialize_wallet(wallet: dict, vials: list):
        """Serialize wallet and vials to Markdown."""
        try:
            data = f"# Neon MCP Export\n\n## Wallet\n- Balance: {wallet['balance']:.4f} $WEBXOS\n- Address: {wallet['address'] or 'none'}\n- User ID: {wallet['user_id'] or 'none'}\n\n## Vials\n"
            for vial in vials:
                data += f"# Vial {vial['id']}\n- Status: {vial['status']}\n- Balance: {vial['balance']:.4f} $WEBXOS\n- Wallet Address: {vial['wallet']['address'] or 'none'}\n\n```python\n{vial['code']}\n```\n---\n"
            return data
        except Exception as e:
            logger.error(f"Serialization failed: {str(e)}")
            raise

    @staticmethod
    def deserialize_wallet(md_content: str):
        """Deserialize Markdown to wallet and vials."""
        try:
            lines = md_content.split('\n')
            wallet = {"balance": 0, "address": None, "user_id": None}
            vials = []
            current_vial = None
            for line in lines:
                if line.startswith("- Balance:") and "Wallet" not in line:
                    wallet["balance"] = float(line.split(":")[1].strip().split(" ")[0])
                elif line.startswith("- Address:"):
                    wallet["address"] = line.split(":")[1].strip() if "none" not in line.lower() else None
                elif line.startswith("- User ID:"):
                    wallet["user_id"] = line.split(":")[1].strip() if "none" not in line.lower() else None
                elif line.startswith("# Vial"):
                    if current_vial:
                        vials.append(current_vial)
                    vial_id = line.split("Vial")[1].strip()
                    current_vial = {"id": vial_id, "status": "Stopped", "balance": 0, "wallet": {"address": None, "balance": 0}, "code": ""}
                elif line.startswith("- Status:") and current_vial:
                    current_vial["status"] = line.split(":")[1].strip()
                elif line.startswith("- Balance:") and current_vial:
                    current_vial["balance"] = float(line.split(":")[1].strip().split(" ")[0])
                    current_vial["wallet"]["balance"] = current_vial["-balance"]
                elif line.startswith("- Wallet Address:") and current_vial:
                    current_vial["wallet"]["address"] = line.split(":")[1].strip() if "none" not in line.lower() else None
                elif line.startswith("```python") and current_vial:
                    code_lines = []
                    for next_line in lines[lines.index(line)+1:]:
                        if next_line.startswith("```"):
                            break
                        code_lines.append(next_line)
                    current_vial["code"] = "\n".join(code_lines)
            if current_vial:
                vials.append(current_vial)
            logger.info("Deserialization successful")
            return wallet, vials
        except Exception as e:
            logger.error(f"Deserialization failed: {str(e)}")
            raise
