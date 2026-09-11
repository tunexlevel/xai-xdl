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
from helper.utils import build_vocab, tokenize_smiles
from helper.dataset import ReactionDataset
from mod.model import Seq2SeqTransformer
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 8
EMB_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 160
EPOCHS = 500
LEARNING_RATE = 0.0006 #6 best
PAD_TOKEN = "<pad>"
DATASET_NAME = "uspto50k_unmapped"
FILE_NAME = f"{DATASET_NAME}_retro_3-3" 
FILE_PATH = "data/raw/uspto50k_unmapped.csv" #ROOT / "data" / "raw" / "train-data" /  f"{DATASET_NAME}.csv"
HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
RETROSYNTHESIS = True  # Set to True for retrosynthesis, False for forward reaction prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_message = "Training Data @ "+str(FILE_NAME)+", Batch Size: "+str(BATCH_SIZE)+", Epochs: "+str(EPOCHS)+", Max Len: "+str(MAX_LEN)+", Emb Dim: "+str(EMB_DIM)+", Hidden Dim: "+str(HIDDEN_DIM)+", Learning Rate: "+str(LEARNING_RATE)

print('' + '=' * len(start_message))
print(start_message)
print('' + '=' * len(start_message))

# Load data
df = load_uspto_file(FILE_PATH)
all_smiles = df['reactants'].tolist() + df['products'].tolist()

# Build vocabulary
token2idx, idx2token = build_vocab(all_smiles)
pad_idx = token2idx[PAD_TOKEN]

# Save vocabulary
with open(ROOT / "tokens" / f"{FILE_NAME}_token2idx.json", "w") as f:
    json.dump(token2idx, f)
with open(ROOT / "tokens" / f"{FILE_NAME}_idx2token.json", "w") as f:
    json.dump(idx2token, f)
    

small_df = df.iloc[:100].copy()


# Dataset & DataLoader
dataset = ReactionDataset(small_df, token2idx, max_len=MAX_LEN, retrosynthesis=RETROSYNTHESIS)
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

# =============================
# Training Loop
# =============================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_reaction_correct = 0
    total_reactions = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for src, tgt in pbar:

        # Move data to device
        src = src.to(device)
        tgt = tgt.to(device)

        # -----------------------------
        # Forward pass
        # -----------------------------

        optimizer.zero_grad()

        # Decoder input:
        # <sos> + reactant tokens
        decoder_input = tgt[:, :-1]

        # Expected output:
        # reactant tokens + <eos>
        target = tgt[:, 1:]

        output = model(src, decoder_input)

        # output shape:
        # [batch_size, target_length, vocab_size]
        #
        # Flatten for CrossEntropyLoss
        output = output.reshape(-1, output.shape[-1])
        target_flat = target.reshape(-1)

        # -----------------------------
        # Loss
        # -----------------------------

        loss = criterion(output, target_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # -----------------------------
        # Token-level accuracy
        # -----------------------------

        preds = output.argmax(dim=1)

        non_pad = target_flat != pad_idx

        correct = (preds == target_flat) & non_pad

        total_correct += correct.sum().item()
        total_tokens += non_pad.sum().item()

        # -----------------------------
        # Reaction-level accuracy
        # -----------------------------

        # Restore:
        # [batch_size, target_length]
        preds_seq = preds.view_as(target)
        target_seq = target

        for i in range(tgt.size(0)):

            # Ignore padding tokens
            valid_positions = target_seq[i] != pad_idx

            pred_reaction = preds_seq[i][valid_positions]
            target_reaction = target_seq[i][valid_positions]

            # Entire reaction sequence must be correct
            if torch.equal(pred_reaction, target_reaction):
                total_reaction_correct += 1

            total_reactions += 1

        # -----------------------------
        # Progress bar
        # -----------------------------

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    # =============================
    # Epoch metrics
    # =============================

    epoch_loss = total_loss / len(dataloader)

    token_accuracy = (
        total_correct / total_tokens
        if total_tokens > 0
        else 0.0
    )

    reaction_accuracy = (
        total_reaction_correct / total_reactions
        if total_reactions > 0
        else 0.0
    )

    print(
        f"\nEpoch {epoch + 1} completed."
        f" Loss: {epoch_loss:.4f}"
        f" | Token Accuracy: {token_accuracy:.4f}"
        f" | Reaction Accuracy: {reaction_accuracy:.4f}"
    )


    # Save model
    torch.save(model.state_dict(), ROOT / "pt" / f"{FILE_NAME}_reaction_model.pt")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")