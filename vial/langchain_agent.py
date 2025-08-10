import torch
import torch.nn as nn
import torch.nn.functional as F
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
import subprocess
import uuid

class NanoGPT(nn.Module):
    def __init__(self, vocab_size=65, n_embd=64, n_head=4, n_layer=4, block_size=32):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(head_size, head_size, bias=False)
        self.query = nn.Linear(head_size, head_size, bias=False)
        self.value = nn.Linear(head_size, head_size, bias=False)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1)
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class LangChainAgent:
    def __init__(self):
        self.nanogpt = NanoGPT()
        self.langchain_runnable = RunnableLambda(self.process_langchain)
        # Load simple checkpoint or train minimally if needed

    def process_langchain(self, input_message):
        if isinstance(input_message, HumanMessage):
            # Use NanoGPT to generate response
            idx = torch.tensor([[0]])  # Dummy input
            generated = self.nanogpt.generate(idx, max_new_tokens=10)
            return {"response": generated.tolist()}
        return {"error": "Invalid input"}

    def handle_message(self, message):
        # Security check if message contains .md
        if '.md' in message and "## WEBXOS Tokenization Tag:" not in message:
            return "Invalid message: Missing WEBXOS Tokenization Tag"
        # Process with NanoGPT
        idx = torch.tensor([[ord(c) % 65 for c in message]])  # Simple tokenization
        generated = self.nanogpt.generate(idx, max_new_tokens=20)
        # Dummy decode
        response = ''.join(chr(g % 128) for g in generated[0].tolist())
        return response

    def handle_git(self, command):
        # Gitbot functionality using subprocess
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return e.stderr

# [xaiartifact: v1.7]
