import torch
import torch.nn as nn
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.searchbot

class NanoGPT(nn.Module):
    def __init__(self, vocab_size=1000, n_embd=128, n_head=4, n_layer=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head), num_layers=n_layer
        )
        self.fc = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

async def train_model():
    model = NanoGPT()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    async for query_doc in db.history.find():
        query = query_doc["query"]
        tokens = tokenize_query(query)
        if len(tokens) < 2: continue
        input_ids = torch.tensor(tokens[:-1]).unsqueeze(0)
        target_ids = torch.tensor(tokens[1:]).unsqueeze(0)
        
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.view(-1, 1000), target_ids.view(-1))
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "nano_gpt.pt")

def tokenize_query(query):
    words = query.lower().split()
    return [hash(word) % 1000 for word in words[:10]]

async def enhance_response(query, apiKey):
    model = NanoGPT()
    try:
        model.load_state_dict(torch.load("nano_gpt.pt"))
    except FileNotFoundError:
        pass
    
    tokens = tokenize_query(query)
    if not tokens: return query
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids)
    enhanced = output.argmax(dim=-1).tolist()[0]
    
    await db.users.update_one(
        {"apiKey": apiKey},
        {"$push": {"query_embeddings": enhanced}}
    )
    return query  # Placeholder: enhance with model output in future
