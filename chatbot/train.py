import torch
import torch.nn as nn
import torch.optim as optim
import json
import pickle
from collections import Counter
import re

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output)
        return output, hidden

# Hyperparameters
VOCAB_SIZE = 5000  # Small vocabulary
HIDDEN_SIZE = 128  # Low-dimensional embeddings
BATCH_SIZE = 32
EPOCHS = 10
MAX_LENGTH = 20  # Max sequence length

# Build vocabulary
def build_vocab(data):
    word_counts = Counter()
    for pair in data:
        words = re.findall(r'\w+', pair['question'].lower()) + re.findall(r'\w+', pair['answer'].lower())
        word_counts.update(words)
    
    # Limit to top VOCAB_SIZE words
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for word, _ in word_counts.most_common(VOCAB_SIZE - len(vocab)):
        vocab[word] = len(vocab)
    
    return vocab

# Tokenize text
def tokenize(text, vocab):
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in re.findall(r'\w+', text.lower())]
    tokens = tokens[:MAX_LENGTH] + [vocab["<EOS>"]]
    while len(tokens) < MAX_LENGTH:
        tokens.append(vocab["<PAD>"])
    return tokens

# Prepare dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Training loop
def train():
    # Load dataset
    data = load_dataset("chatbot/data/webxos_qa.json")
    
    # Build and save vocabulary
    vocab = build_vocab(data)
    with open("chatbot/model/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    # Initialize model
    encoder = Encoder(len(vocab), HIDDEN_SIZE)
    decoder = Decoder(HIDDEN_SIZE, len(vocab))
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    
    # Prepare data
    inputs = [tokenize(pair['question'], vocab) for pair in data]
    targets = [tokenize(pair['answer'], vocab) for pair in data]
    
    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(0, len(data), BATCH_SIZE):
            batch_inputs = torch.tensor(inputs[i:i+BATCH_SIZE], dtype=torch.long)
            batch_targets = torch.tensor(targets[i:i+BATCH_SIZE], dtype=torch.long)
            
            optimizer.zero_grad()
            _, hidden = encoder(batch_inputs)
            
            # Decoder input starts with <SOS>
            decoder_input = torch.tensor([[vocab["<SOS>"]] * len(batch_inputs)], dtype=torch.long)
            decoder_hidden = hidden
            
            loss = 0
            for t in range(MAX_LENGTH):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output.squeeze(1), batch_targets[:, t])
                decoder_input = batch_targets[:, t:t+1]  # Teacher forcing
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / MAX_LENGTH
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / (len(data) // BATCH_SIZE)}")
    
    # Quantize model to reduce size
    encoder.eval()
    decoder.eval()
    encoder = torch.quantization.quantize_dynamic(encoder, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    decoder = torch.quantization.quantize_dynamic(decoder, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    
    # Save model
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }, "chatbot/model/chatbot_model.pt")

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    import os
    os.makedirs("chatbot/model", exist_ok=True)
    os.makedirs("chatbot/data", exist_ok=True)
    train()