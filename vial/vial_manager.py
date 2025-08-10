import json
import uuid

def initialize_vials():
    return [
        {
            "id": f"vial{i+1}",
            "status": "stopped",
            "code": "import torch\nimport torch.nn as nn\n\nclass VialAgent(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 1)\n    def forward(self, x):\n        return torch.sigmoid(self.fc(x))\n\nmodel = VialAgent()",
            "codeLength": 0,
            "isPython": True,
            "webxosHash": str(uuid.uuid4()),
            "wallet": {"address": None, "balance": 0.0},
            "tasks": []
        }
        for i in range(4)
    ]

def export_vial(vial, filename):
    with open(f"{filename}.md", 'w') as f:
        f.write(
            f"# Vial Agent: {vial['id']}\n"
            f"- Status: {vial['status']}\n"
            f"- Language: {'Python' if vial['isPython'] else 'JavaScript'}\n"
            f"- Code Length: {vial['codeLength']} bytes\n"
            f"- $WEBXOS Hash: {vial['webxosHash']}\n"
            f"- Wallet Balance: {vial['wallet']['balance']:.4f} $WEBXOS\n"
            f"- Wallet Address: {vial['wallet']['address'] or 'none'}\n"
            f"- Tasks: {', '.join(vial['tasks']) or 'none'}\n"
            f"```{'python' if vial['isPython'] else 'javascript'}\n{vial['code']}\n```"
        )

def import_vials(filename):
    vials = []
    with open(filename, 'r') as f:
        content = f.read()
        sections = content.split('---\n\n')
        for section in sections:
            if section.startswith('# Vial Agent:'):
                vial = {}
                lines = section.split('\n')
                vial['id'] = lines[0].replace('# Vial Agent: ', '').strip()
                for line in lines[1:]:
                    if line.startswith('- Status:'):
                        vial['status'] = line.replace('- Status: ', '').strip()
                    elif line.startswith('- Language:'):
                        vial['isPython'] = line.replace('- Language: ', '').strip() == 'Python'
                    elif line.startswith('- Code Length:'):
                        vial['codeLength'] = int(line.replace('- Code Length: ', '').replace(' bytes', '').strip())
                    elif line.startswith('- $WEBXOS Hash:'):
                        vial['webxosHash'] = line.replace('- $WEBXOS Hash: ', '').strip()
                    elif line.startswith('- Wallet Balance:'):
                        vial['wallet'] = {"balance": float(line.replace('- Wallet Balance: ', '').replace(' $WEBXOS', '').strip())}
                    elif line.startswith('- Wallet Address:'):
                        vial['wallet']['address'] = line.replace('- Wallet Address: ', '').strip()
                    elif line.startswith('- Tasks:'):
                        vial['tasks'] = line.replace('- Tasks: ', '').strip().split(', ') if line.strip() != '- Tasks: none' else []
                    elif line.startswith('```'):
                        code_lines = []
                        in_code = False
                        for code_line in lines[lines.index(line)+1:]:
                            if code_line.startswith('```'):
                                break
                            code_lines.append(code_line)
                        vial['code'] = '\n'.join(code_lines)
                vials.append(vial)
    return vials
