import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from data_loader import load_uspto_file
from utils import build_vocab
from dataset import ReactionDataset
from model_1 import Seq2SeqTransformer
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
EMB_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 120
EPOCHS = 20
LEARNING_RATE = 1e-3
PAD_TOKEN = "<pad>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = load_uspto_file("raw_train.csv", max_samples=50000)
all_smiles = df['reactants'].tolist() + df['products'].tolist()

# Build vocabulary
token2idx, idx2token = build_vocab(all_smiles)
pad_idx = token2idx[PAD_TOKEN]

# Save vocab
with open("token2idx.json", "w") as f:
    json.dump(token2idx, f)
with open("idx2token.json", "w") as f:
    json.dump(idx2token, f)

# Dataset & DataLoader
dataset = ReactionDataset(df, token2idx, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = Seq2SeqTransformer(
    input_dim=len(token2idx),
    output_dim=len(token2idx),
    emb_dim=EMB_DIM,
    nhead=8,                 # New param: Heads
    num_encoder_layers=3,    # New param: Depth
    num_decoder_layers=3,
    dim_feedforward=HIDDEN_DIM,
    pad_idx=pad_idx
).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # remove <eos> for input
        output = output.reshape(-1, output.shape[-1])
        target = tgt[:, 1:].reshape(-1)  # shift target to remove <sos>

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        preds = output.argmax(dim=1)
        correct = (preds == target) & (target != pad_idx)
        total_correct += correct.sum().item()
        total_tokens += (target != pad_idx).sum().item()

        pbar.set_postfix(loss=loss.item())

    epoch_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens
    print(f"\nEpoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    if accuracy == 1.0:
        print("Perfect accuracy achieved, stopping training.")
        break
# Save model
torch.save(model.state_dict(), "reaction_model.pt")
