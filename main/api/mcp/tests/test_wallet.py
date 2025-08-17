import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from tools.wallet import WalletTool
from config.config import DatabaseConfig
from unittest.mock import AsyncMock, patch
import hashlib
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_db():
    db = AsyncMock(spec=DatabaseConfig)
    db.query = AsyncMock()
    return db

@pytest.mark.asyncio
async def test_wallet_get_vial_balance_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 42547.0}] })
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.getVialBalance",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["vial_id"] == "vial1"
    assert result["balance"] == 10636.75  # 42547.0 / 4

@pytest.mark.asyncio
async def test_wallet_import_example_md(client, mock_db):
    markdown = """# WebXOS Vial and Wallet Export

## Agentic Network
- Network ID: 54965687-3871-4f3d-a803-ac9840af87c4
- Session Start: 2025-08-17T16:58:26.018Z
- Session Duration: 0.00 seconds
- Reputation: 1200987188

## Wallet
- Wallet Key: a1d57580-d88b-4c90-a0f8-6f2c8511b1e4
- Session Balance: 42547.0000 $WEBXOS
- Address: e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d
- Hash: 16f11f690efece2c2a76a9ef419366522bc8a7497b6aa0e4e5c150faab1dfcb8

## API Credentials
- Key: 8d0ff3fc-65ae-4e91-b0da-7fa272b37912
- Secret: ac44567620546a74d06b4bc58230b92f689168f16a1879d2f06fa01456c57ed4

## Blockchain
- Blocks: 1958
- Last Hash: 16f11f690efece2c2a76a9ef419366522bc8a7497b6aa0e4e5c150faab1dfcb8

## Vials
# Vial Agent: vial1
- Status: running
- Language: Python
- Code Length: 586372 bytes
- $WEBXOS Hash: 1ea34723-38db-45a3-b5bb-cb07b383c2fd
- Wallet Balance: 10636.7500 $WEBXOS
- Wallet Address: 59cc28fa-ff12-4ad0-bb4e-af64fe2d441d
- Wallet Hash: 16f11f690efece2c2a76a9ef419366522bc8a7497b6aa0e4e5c150faab1dfcb8
- Tasks: none
- Quantum State: {
      "qubits": [],
      "entanglement": "synced"
    }
- Training Data: [
      {
        "tasks": [],
        "parameters": {},
        "hash": "886ec7d3bd933f76fa5a15d4babb98b92a2235a814afde714fea91bbae04bbbf"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "66cb438f9e70d8b6a70c4ea81e3cf97b8eb95beee50a87388ff44887e77eeeca"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "7d1c2ab735cf65cbaf377be2f15cf31106a47c9b9bcf2e2a6d1a4121c423fe27"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "853b8f7cc661fd56c9c1cec4ae86d1cdf4890864da73e8397a5843e9ae86e47f"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "033f357083c958bbb0168f8a6f6761c5673ffdd038b2a090e39412b245d4a8cb"
      }
    ]
- Config: {}

```python
import torch
import torch.nn as nn

class VialAgent1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = VialAgent1()
```

---

# Vial Agent: vial2
- Status: running
- Language: Python
- Code Length: 586369 bytes
- $WEBXOS Hash: 2983fb73-2ce0-402f-8b76-f20175302fc4
- Wallet Balance: 10636.7500 $WEBXOS
- Wallet Address: e7fa09d2-9d98-4cfd-bc44-4f8035e1cf20
- Wallet Hash: 16f11f690efece2c2a76a9ef419366522bc8a7497b6aa0e4e5c150faab1dfcb8
- Tasks: none
- Quantum State: {
      "qubits": [],
      "entanglement": "synced"
    }
- Training Data: [
      {
        "tasks": [],
        "parameters": {},
        "hash": "886ec7d3bd933f76fa5a15d4babb98b92a2235a814afde714fea91bbae04bbbf"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "66cb438f9e70d8b6a70c4ea81e3cf97b8eb95beee50a87388ff44887e77eeeca"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "7d1c2ab735cf65cbaf377be2f15cf31106a47c9b9bcf2e2a6d1a4121c423fe27"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "853b8f7cc661fd56c9c1cec4ae86d1cdf4890864da73e8397a5843e9ae86e47f"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "033f357083c958bbb0168f8a6f6761c5673ffdd038b2a090e39412b245d4a8cb"
      }
    ]
- Config: {}

```python
import torch
import torch.nn as nn

class VialAgent2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 2)
    def forward(self, x):
        return torch.tanh(self.fc(x))

model = VialAgent2()
```

---

# Vial Agent: vial3
- Status: running
- Language: Python
- Code Length: 586374 bytes
- $WEBXOS Hash: 7d0b10c0-c3ef-4e09-8f83-0e9a9d3eabf8
- Wallet Balance: 10636.7500 $WEBXOS
- Wallet Address: af1baa76-5a1f-4b9b-9463-55ced39fe6dd
- Wallet Hash: 16f11f690efece2c2a76a9ef419366522bc8a7497b6aa0e4e5c150faab1dfcb8
- Tasks: none
- Quantum State: {
      "qubits": [],
      "entanglement": "synced"
    }
- Training Data: [
      {
        "tasks": [],
        "parameters": {},
        "hash": "886ec7d3bd933f76fa5a15d4babb98b92a2235a814afde714fea91bbae04bbbf"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "66cb438f9e70d8b6a70c4ea81e3cf97b8eb95beee50a87388ff44887e77eeeca"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "7d1c2ab735cf65cbaf377be2f15cf31106a47c9b9bcf2e2a6d1a4121c423fe27"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "853b8f7cc661fd56c9c1cec4ae86d1cdf4890864da73e8397a5843e9ae86e47f"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "033f357083c958bbb0168f8a6f6761c5673ffdd038b2a090e39412b245d4a8cb"
      }
    ]
- Config: {}

```python
import torch
import torch.nn as nn

class VialAgent3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(15, 3)
    def forward(self, x):
        return torch.tanh(self.fc(x))

model = VialAgent3()
```

---

# Vial Agent: vial4
- Status: running
- Language: Python
- Code Length: 586379 bytes
- $WEBXOS Hash: 405e6101-2b0f-4aad-851e-7e5710994fb2
- Wallet Balance: 10636.7500 $WEBXOS
- Wallet Address: 143ec9b7-0e80-4842-9f03-e936da7a5844
- Wallet Hash: 16f11f690efece2c2a76a9ef419366522bc8a7497b6aa0e4e5c150faab1dfcb8
- Tasks: none
- Quantum State: {
      "qubits": [],
      "entanglement": "synced"
    }
- Training Data: [
      {
        "tasks": [],
        "parameters": {},
        "hash": "886ec7d3bd933f76fa5a15d4babb98b92a2235a814afde714fea91bbae04bbbf"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "66cb438f9e70d8b6a70c4ea81e3cf97b8eb95beee50a87388ff44887e77eeeca"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "7d1c2ab735cf65cbaf377be2f15cf31106a47c9b9bcf2e2a6d1a4121c423fe27"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "853b8f7cc661fd56c9c1cec4ae86d1cdf4890864da73e8397a5843e9ae86e47f"
      },
      {
        "tasks": [],
        "parameters": {},
        "hash": "033f357083c958bbb0168f8a6f6761c5673ffdd038b2a090e39412b245d4a8cb"
      }
    ]
- Config: {}

```python
import torch
import torch.nn as nn

class VialAgent4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(25, 4)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

model = VialAgent4()
```

## Instructions
- **Reuse**: Import this .md file via the "Import" button to resume training.
- **Extend**: Modify agent code externally, then reimport.
- **Share**: Send this .md file to others to continue training with the same wallet.
- **API**: Use API credentials with LangChain to train vials (online mode only).
- **Cash Out**: $WEBXOS balance and reputation are tied to the wallet address and hash for secure verification (online mode only).

Generated by Vial MCP Controller
"""
    hash_value = "16f11f690efece2c2a76a9ef419366522bc8a7497b6aa0e4e5c150faab1dfcb8"
    
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] }),  # User exists
        type("Result", (), {"rows": [{}]} )  # Balance updated
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.importWallet",
        "params": {
            "user_id": "user_12345",
            "markdown": markdown,
            "hash": hash_value
        },
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["imported_vials"] == ["vial1", "vial2", "vial3", "vial4"]
    assert result["total_balance"] == 42647.0  # 100.0 + 42547.0

@pytest.mark.asyncio
async def test_wallet_import_invalid_hash(client, mock_db):
    markdown = "Invalid markdown"
    hash_value = hashlib.sha256("Different markdown".encode()).hexdigest()
    
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] })
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.importWallet",
        "params": {
            "user_id": "user_12345",
            "markdown": markdown,
            "hash": hash_value
        },
        "id": 1
    })
    
    assert response.status_code == 400
    assert "Hash mismatch" in response.json()["error"]["message"]

@pytest.mark.asyncio
async def test_wallet_export_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 42547.0, "wallet_address": "e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d", "api_key": "8d0ff3fc-65ae-4e91-b0da-7fa272b37912", "api_secret": "ac44567620546a74d06b4bc58230b92f689168f16a1879d2f06fa01456c57ed4"}] })
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.exportVials",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert "# WebXOS Vial and Wallet Export" in result["markdown"]
    assert "Wallet Balance: 10636.7500 $WEBXOS" in result["markdown"]
    assert "Quantum State: {\n      \"qubits\": [],\n      \"entanglement\": \"synced\"\n    }" in result["markdown"]
    assert "hash" in result

@pytest.mark.asyncio
async def test_wallet_mine_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 42547.0}] }),  # User exists
        type("Result", (), {"rows": [{}]} )  # Balance updated
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.mineVial",
        "params": {"user_id": "user_12345", "vial_id": "vial1", "nonce": 12345},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert "hash" in result
    assert "reward" in result
    assert "balance" in result

@pytest.mark.asyncio
async def test_wallet_batch_sync_success(client, mock_db):
    markdown = "class VialAgent1:\n    def __init__(self):\n        self.balance = 50.0"
    hash_value = hashlib.sha256(markdown.encode()).hexdigest()
    
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 42547.0}] }),  # User exists
        type("Result", (), {"rows": [{}]} )  # Balance updated
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.batchSync",
        "params": {
            "user_id": "user_12345",
            "operations": [
                {
                    "method": "importWallet",
                    "markdown": markdown,
                    "hash": hash_value
                },
                {
                    "method": "mineVial",
                    "vial_id": "vial1",
                    "nonce": 12345
                }
            ]
        },
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result["results"]) == 2
    assert result["results"][0]["total_balance"] == 42597.0
    assert "hash" in result["results"][1]

@pytest.mark.asyncio
async def test_wallet_quantum_link_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345"}] }),
        type("Result", (), {"rows": [{}]} )  # Link inserted
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.quantumLink",
        "params": {"user_id": "user_12345"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert "link_id" in result
