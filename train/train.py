import os
import sys
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from helper.data_loader import load_uspto_file
from helper.utils import build_vocab
from helper.dataset import ReactionDataset
from mod.model import Seq2SeqTransformer
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
EMB_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 120
EPOCHS = 20
LEARNING_RATE = 1e-3
PAD_TOKEN = "<pad>"
FILE_PATH = ROOT / "data" /  "uspto50k_unmapped.csv"
HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = load_uspto_file(FILE_PATH, max_samples=50000)
all_smiles = df['reactants'].tolist() + df['products'].tolist()

# Build vocabulary
token2idx, idx2token = build_vocab(all_smiles)
pad_idx = token2idx[PAD_TOKEN]

# Save vocabulary
with open(ROOT / "tokens" / "token2idx.json", "w") as f:
    json.dump(token2idx, f)
with open(ROOT / "tokens" / "idx2token.json", "w") as f:
    json.dump(idx2token, f)
    

# Dataset & DataLoader
dataset = ReactionDataset(df, token2idx, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = Seq2SeqTransformer(
    input_dim=len(token2idx),
    output_dim=len(token2idx),
    emb_dim=EMB_DIM,
    nhead=HEADS,                 # New param: Heads
    num_encoder_layers=NUM_ENCODER_LAYERS,    # New param: Depth
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=HIDDEN_DIM,
    pad_idx=pad_idx
).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


start_time = time.time()

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    total_reaction_correct = 0
    total_reactions = 0

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
        
    
    # -----------------------------
    # Reaction-level accuracy
    # -----------------------------
    preds_seq = preds.view(tgt[:, 1:].shape)
    target_seq = target.view(tgt[:, 1:].shape)
    
    for i in range(tgt.size(0)):

        # Ignore padding
        valid_positions = target_seq[i] != pad_idx

        pred_reaction = preds_seq[i][valid_positions]
        target_reaction = target_seq[i][valid_positions]

        # Entire sequence must be correct
        if torch.equal(pred_reaction, target_reaction):
            total_reaction_correct += 1

        total_reactions += 1
        
        
    
    
    preds_seq = preds.view(tgt[:, 1:].shape)
    target_seq = target.view(tgt[:, 1:].shape)
    
    reaction_accuracy = (
    total_reaction_correct / total_reactions
    if total_reactions > 0
        else 0.0
    )


    print(f"\nEpoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}\n", 
          f"Reaction-level Accuracy: {reaction_accuracy:.4f}")

    if accuracy == 1.0:
        print("Perfect accuracy achieved, stopping training.")
        break
# Save model
torch.save(model.state_dict(), ROOT / "pt" / "reaction_model.pt")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")
