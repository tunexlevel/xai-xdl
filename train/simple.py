import os
import sys
import json
from pathlib import Path
import time

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from helper.data_loader import load_uspto_file
from helper.utils import build_vocab
from helper.dataset import ReactionDataset
from mod.model import Seq2SeqTransformer


# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 32

EMB_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 120

EPOCHS = 20
LEARNING_RATE = 5e-4

PAD_TOKEN = "<pad>"

DATASET_NAME = "uspto_mit_mapped"

FILE_NAME = f"{DATASET_NAME}_ed_4-4"

FILE_PATH = "/content/drive/MyDrive/Colab Notebooks/uspto_mit_mapped.csv"

HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CHECKPOINT CONFIGURATION
# ============================================================

# IMPORTANT:
# Store checkpoints on Google Drive so they survive
# Colab runtime disconnections/resets.

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/ChemXAI/checkpoints"
) / FILE_NAME

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"


# ============================================================
# TRAINING INFORMATION
# ============================================================

start_message = (
    f"Training Data @ {FILE_NAME}, "
    f"Batch Size: {BATCH_SIZE}, "
    f"Epochs: {EPOCHS}, "
    f"Max Len: {MAX_LEN}, "
    f"Emb Dim: {EMB_DIM}, "
    f"Hidden Dim: {HIDDEN_DIM}, "
    f"Learning Rate: {LEARNING_RATE}"
)

print("=" * len(start_message))
print(start_message)
print("=" * len(start_message))


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = load_uspto_file(FILE_PATH)

print(f"Number of reactions loaded: {len(df)}")

all_smiles = (
    df["reactants"].tolist()
    + df["products"].tolist()
)


# ============================================================
# BUILD VOCABULARY
# ============================================================

print("Building vocabulary...")

token2idx, idx2token = build_vocab(all_smiles)

pad_idx = token2idx[PAD_TOKEN]

print(f"Vocabulary size: {len(token2idx)}")


# ============================================================
# SAVE VOCABULARY
# ============================================================

TOKEN_DIR = ROOT / "tokens"
TOKEN_DIR.mkdir(parents=True, exist_ok=True)

with open(
    TOKEN_DIR / f"{FILE_NAME}_token2idx.json",
    "w"
) as f:
    json.dump(token2idx, f)

with open(
    TOKEN_DIR / f"{FILE_NAME}_idx2token.json",
    "w"
) as f:
    json.dump(idx2token, f)


# ============================================================
# DATASET & DATALOADER
# ============================================================

dataset = ReactionDataset(
    df,
    token2idx,
    max_len=MAX_LEN
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")


# ============================================================
# MODEL
# ============================================================

model = Seq2SeqTransformer(
    input_dim=len(token2idx),
    output_dim=len(token2idx),
    emb_dim=EMB_DIM,
    nhead=HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=HIDDEN_DIM,
    pad_idx=pad_idx
).to(device)


# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT VARIABLES
# ============================================================

start_epoch = 0

best_loss = float("inf")

training_history = []


# ============================================================
# LOAD LATEST CHECKPOINT IF AVAILABLE
# ============================================================

if LATEST_CHECKPOINT.exists():

    print("\n" + "=" * 60)
    print("CHECKPOINT FOUND")
    print("=" * 60)

    print(f"Loading: {LATEST_CHECKPOINT}")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_loss = checkpoint.get(
        "best_loss",
        float("inf")
    )

    training_history = checkpoint.get(
        "training_history",
        []
    )

    print(
        f"Resuming from epoch {start_epoch + 1}"
    )

    print(
        f"Previous best loss: {best_loss:.6f}"
    )

    print("=" * 60)

else:

    print("\nNo checkpoint found.")
    print("Starting training from epoch 1.")


# ============================================================
# TRAINING
# ============================================================

start_time = time.time()


for epoch in range(start_epoch, EPOCHS):

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


    # ========================================================
    # BATCH TRAINING
    # ========================================================

    for src, tgt in pbar:

        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Remove <eos> from decoder input
        output = model(
            src,
            tgt[:, :-1]
        )

        output = output.reshape(
            -1,
            output.shape[-1]
        )

        # Shift target to remove <sos>
        target = tgt[:, 1:].reshape(-1)

        # Loss
        loss = criterion(
            output,
            target
        )

        loss.backward()

        optimizer.step()


        # ====================================================
        # LOSS
        # ====================================================

        total_loss += loss.item()


        # ====================================================
        # TOKEN ACCURACY
        # ====================================================

        preds = output.argmax(dim=1)

        valid = target != pad_idx

        correct = (
            (preds == target)
            & valid
        )

        total_correct += (
            correct.sum().item()
        )

        total_tokens += (
            valid.sum().item()
        )


        # ====================================================
        # REACTION-LEVEL ACCURACY
        # ====================================================

        preds_seq = preds.view(
            tgt[:, 1:].shape
        )

        target_seq = target.view(
            tgt[:, 1:].shape
        )

        for i in range(tgt.size(0)):

            valid_positions = (
                target_seq[i] != pad_idx
            )

            pred_reaction = (
                preds_seq[i][valid_positions]
            )

            target_reaction = (
                target_seq[i][valid_positions]
            )

            if torch.equal(
                pred_reaction,
                target_reaction
            ):
                total_reaction_correct += 1

            total_reactions += 1


        # ====================================================
        # PROGRESS BAR
        # ====================================================

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    # ========================================================
    # EPOCH METRICS
    # ========================================================

    epoch_loss = (
        total_loss / len(dataloader)
    )

    accuracy = (
        total_correct / total_tokens
        if total_tokens > 0
        else 0.0
    )

    reaction_accuracy = (
        total_reaction_correct
        / total_reactions
        if total_reactions > 0
        else 0.0
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"\nEpoch {epoch + 1} completed."
    )

    print(
        f"Loss: {epoch_loss:.6f}"
    )

    print(
        f"Token Accuracy: {accuracy:.6f}"
    )

    print(
        f"Reaction-level Accuracy: "
        f"{reaction_accuracy:.6f}"
    )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    epoch_record = {
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "token_accuracy": accuracy,
        "reaction_accuracy": reaction_accuracy
    }

    training_history.append(
        epoch_record
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "loss": epoch_loss,

        "token_accuracy":
            accuracy,

        "reaction_accuracy":
            reaction_accuracy,

        "best_loss":
            min(best_loss, epoch_loss),

        "training_history":
            training_history,

        # Save configuration too
        # so we know exactly how this model was trained.

        "config": {
            "dataset_name": DATASET_NAME,
            "file_name": FILE_NAME,
            "batch_size": BATCH_SIZE,
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_len": MAX_LEN,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "heads": HEADS,
            "num_encoder_layers":
                NUM_ENCODER_LAYERS,
            "num_decoder_layers":
                NUM_DECODER_LAYERS,
            "vocab_size":
                len(token2idx)
        }
    }


    # ========================================================
    # SAVE LATEST CHECKPOINT
    # ========================================================

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT
    )

    print(
        f"Latest checkpoint saved: "
        f"{LATEST_CHECKPOINT}"
    )


    # ========================================================
    # SAVE EPOCH CHECKPOINT
    # ========================================================

    epoch_checkpoint = (
        CHECKPOINT_DIR
        / f"epoch_{epoch + 1:03d}.pt"
    )

    torch.save(
        checkpoint,
        epoch_checkpoint
    )

    print(
        f"Epoch checkpoint saved: "
        f"{epoch_checkpoint}"
    )


    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        # Update best_loss inside checkpoint
        checkpoint["best_loss"] = best_loss

        torch.save(
            checkpoint,
            BEST_CHECKPOINT
        )

        print(
            f"New BEST model saved! "
            f"Loss: {best_loss:.6f}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

PT_DIR = ROOT / "pt"
PT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = (
    PT_DIR
    / f"{FILE_NAME}_reaction_model.pt"
)

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

HISTORY_PATH = (
    CHECKPOINT_DIR
    / "training_history.json"
)

with open(
    HISTORY_PATH,
    "w"
) as f:
    json.dump(
        training_history,
        f,
        indent=4
    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

end_time = time.time()

elapsed = (
    end_time - start_time
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Total training time: "
    f"{elapsed / 3600:.2f} hours"
)

print(
    f"Final model: "
    f"{FINAL_MODEL_PATH}"
)

print(
    f"Latest checkpoint: "
    f"{LATEST_CHECKPOINT}"
)

print(
    f"Best checkpoint: "
    f"{BEST_CHECKPOINT}"
)

print(
    f"Training history: "
    f"{HISTORY_PATH}"
)

print("=" * 60)
