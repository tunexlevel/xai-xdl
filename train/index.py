import os
import sys
import json
import time
from pathlib import Path

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

# Google Drive dataset
FILE_PATH = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "uspto_mit_mapped.csv"
)

HEADS = 8

NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA version: {torch.version.cuda}"
    )

    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("=" * 70)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

# IMPORTANT:
# These files are stored on Google Drive.
# Therefore they survive Colab runtime disconnections.

CHECKPOINT_DIR = (
    Path(
        "/content/drive/MyDrive/Colab Notebooks/"
        "ChemXAI/checkpoints"
    )
    / FILE_NAME
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECKPOINT FILES
# ============================================================

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR / "latest.pt"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best.pt"
)

HISTORY_FILE = (
    CHECKPOINT_DIR / "training_history.json"
)


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

print()
print("=" * len(start_message))
print(start_message)
print("=" * len(start_message))


# ============================================================
# CHECK DATASET EXISTS
# ============================================================

if not os.path.exists(FILE_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{FILE_PATH}\n\n"
        "Make sure Google Drive is mounted."
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

print(f"Dataset: {FILE_PATH}")

df = load_uspto_file(FILE_PATH)

print(
    f"Number of reactions loaded: {len(df):,}"
)

print(
    f"Columns: {list(df.columns)}"
)


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = [
    "reactants",
    "products"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found in dataset."
        )


print(
    f"Reactant examples: "
    f"{df['reactants'].head(2).tolist()}"
)

print(
    f"Product examples: "
    f"{df['products'].head(2).tolist()}"
)


# ============================================================
# BUILD VOCABULARY
# ============================================================

print()
print("=" * 70)
print("BUILDING VOCABULARY")
print("=" * 70)

all_smiles = (
    df["reactants"].tolist()
    +
    df["products"].tolist()
)

token2idx, idx2token = build_vocab(
    all_smiles
)

pad_idx = token2idx[PAD_TOKEN]

print(
    f"Vocabulary size: {len(token2idx):,}"
)

print(
    f"PAD index: {pad_idx}"
)


# ============================================================
# SAVE VOCABULARY
# ============================================================

TOKEN_DIR = ROOT / "tokens"

TOKEN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOKEN2IDX_PATH = (
    TOKEN_DIR
    / f"{FILE_NAME}_token2idx.json"
)

IDX2TOKEN_PATH = (
    TOKEN_DIR
    / f"{FILE_NAME}_idx2token.json"
)

with open(
    TOKEN2IDX_PATH,
    "w"
) as f:

    json.dump(
        token2idx,
        f,
        indent=2
    )


with open(
    IDX2TOKEN_PATH,
    "w"
) as f:

    json.dump(
        idx2token,
        f,
        indent=2
    )

print(
    f"Vocabulary saved to: {TOKEN_DIR}"
)


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 70)
print("CREATING DATASET")
print("=" * 70)

dataset = ReactionDataset(
    df,
    token2idx,
    max_len=MAX_LEN
)

print(
    f"Dataset size: {len(dataset):,}"
)


# ============================================================
# DATALOADER
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(
    f"Number of batches per epoch: "
    f"{len(dataloader):,}"
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)

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
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT STATE
# ============================================================

# Number of the last COMPLETED epoch.
#
# Example:
#
# completed_epoch = 4
#
# means:
#
# Epochs 1, 2, 3 and 4 are complete.
#
# The next training run starts Epoch 5.

completed_epoch = 0

best_loss = float("inf")

training_history = []


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if LATEST_CHECKPOINT.exists():

    print()
    print("=" * 70)
    print("CHECKPOINT FOUND")
    print("=" * 70)

    print(
        f"Checkpoint: {LATEST_CHECKPOINT}"
    )

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device
    )

    # --------------------------------------------------------
    # Restore model
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Restore optimizer
    # --------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # --------------------------------------------------------
    # Restore completed epoch
    # --------------------------------------------------------

    completed_epoch = checkpoint.get(
        "completed_epoch",
        checkpoint.get("epoch", 0)
    )

    # --------------------------------------------------------
    # Restore best loss
    # --------------------------------------------------------

    best_loss = checkpoint.get(
        "best_loss",
        float("inf")
    )

    # --------------------------------------------------------
    # Restore history
    # --------------------------------------------------------

    training_history = checkpoint.get(
        "training_history",
        []
    )

    print(
        f"Last completed epoch: "
        f"{completed_epoch}"
    )

    print(
        f"Best loss: "
        f"{best_loss:.6f}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # If the checkpoint says epoch 4 is complete,
    # we start epoch 5.
    #
    # If Colab died during epoch 5, epoch 4 remains
    # the last completed epoch, so epoch 5 starts again.
    # --------------------------------------------------------

    start_epoch = completed_epoch

    print(
        f"Next epoch to train: "
        f"{start_epoch + 1}"
    )

    print("=" * 70)

else:

    print()
    print("=" * 70)
    print("NO CHECKPOINT FOUND")
    print("=" * 70)

    print(
        "Starting training from Epoch 1."
    )

    start_epoch = 0


# ============================================================
# TRAINING START
# ============================================================

start_time = time.time()


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    start_epoch,
    EPOCHS
):

    current_epoch = epoch + 1

    print()
    print("=" * 70)

    print(
        f"STARTING EPOCH "
        f"{current_epoch}/{EPOCHS}"
    )

    print("=" * 70)

    model.train()

    total_loss = 0.0

    total_correct = 0
    total_tokens = 0

    total_reaction_correct = 0
    total_reactions = 0


    # ========================================================
    # PROGRESS BAR
    # ========================================================

    pbar = tqdm(
        dataloader,
        desc=(
            f"Epoch "
            f"{current_epoch}/{EPOCHS}"
        )
    )


    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_idx, (src, tgt) in enumerate(
        pbar,
        start=1
    ):

        src = src.to(device)
        tgt = tgt.to(device)


        # ----------------------------------------------------
        # RESET GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        output = model(
            src,
            tgt[:, :-1]
        )


        # ----------------------------------------------------
        # RESHAPE OUTPUT
        # ----------------------------------------------------

        output = output.reshape(
            -1,
            output.shape[-1]
        )


        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target = tgt[:, 1:].reshape(-1)


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            output,
            target
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # ACCUMULATE LOSS
        # ----------------------------------------------------

        total_loss += loss.item()


        # ----------------------------------------------------
        # TOKEN ACCURACY
        # ----------------------------------------------------

        preds = output.argmax(
            dim=1
        )

        valid = (
            target != pad_idx
        )

        correct = (
            (preds == target)
            &
            valid
        )

        total_correct += (
            correct.sum().item()
        )

        total_tokens += (
            valid.sum().item()
        )


        # ----------------------------------------------------
        # REACTION-LEVEL ACCURACY
        # ----------------------------------------------------

        preds_seq = preds.view(
            tgt[:, 1:].shape
        )

        target_seq = target.view(
            tgt[:, 1:].shape
        )

        for i in range(
            tgt.size(0)
        ):

            valid_positions = (
                target_seq[i]
                != pad_idx
            )

            pred_reaction = (
                preds_seq[i][
                    valid_positions
                ]
            )

            target_reaction = (
                target_seq[i][
                    valid_positions
                ]
            )

            if torch.equal(
                pred_reaction,
                target_reaction
            ):

                total_reaction_correct += 1

            total_reactions += 1


        # ----------------------------------------------------
        # PROGRESS INFORMATION
        # ----------------------------------------------------

        current_token_accuracy = (
            total_correct
            /
            total_tokens
            if total_tokens > 0
            else 0.0
        )

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_token_accuracy:.4f}"
        )


    # ========================================================
    # EPOCH METRICS
    # ========================================================

    epoch_loss = (
        total_loss
        /
        len(dataloader)
    )

    accuracy = (
        total_correct
        /
        total_tokens
        if total_tokens > 0
        else 0.0
    )

    reaction_accuracy = (
        total_reaction_correct
        /
        total_reactions
        if total_reactions > 0
        else 0.0
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print()
    print(
        f"Epoch {current_epoch} completed."
    )

    print(
        f"Loss: "
        f"{epoch_loss:.6f}"
    )

    print(
        f"Token Accuracy: "
        f"{accuracy:.6f}"
    )

    print(
        f"Reaction-level Accuracy: "
        f"{reaction_accuracy:.6f}"
    )


    # ========================================================
    # UPDATE TRAINING HISTORY
    # ========================================================

    epoch_record = {

        "epoch":
            current_epoch,

        "loss":
            epoch_loss,

        "token_accuracy":
            accuracy,

        "reaction_accuracy":
            reaction_accuracy
    }

    training_history.append(
        epoch_record
    )


    # ========================================================
    # DETERMINE BEST MODEL
    # ========================================================

    is_best = (
        epoch_loss < best_loss
    )

    if is_best:

        best_loss = epoch_loss

        print()
        print(
            "NEW BEST MODEL!"
        )

        print(
            f"Best loss: "
            f"{best_loss:.6f}"
        )


    # ========================================================
    # IMPORTANT CHECKPOINT
    #
    # We only mark this epoch as completed AFTER the entire
    # epoch has finished successfully.
    #
    # Therefore, if Colab crashes halfway through this epoch,
    # the previous checkpoint remains the last completed epoch.
    # ========================================================

    completed_epoch = current_epoch


    # ========================================================
    # CREATE CHECKPOINT
    # ========================================================

    checkpoint = {

        # ----------------------------------------------------
        # TRAINING POSITION
        # ----------------------------------------------------

        "completed_epoch":
            completed_epoch,

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        "model_state_dict":
            model.state_dict(),

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        "optimizer_state_dict":
            optimizer.state_dict(),

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        "loss":
            epoch_loss,

        "token_accuracy":
            accuracy,

        "reaction_accuracy":
            reaction_accuracy,

        "best_loss":
            best_loss,

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        "training_history":
            training_history,

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        "config": {

            "dataset_name":
                DATASET_NAME,

            "file_name":
                FILE_NAME,

            "batch_size":
                BATCH_SIZE,

            "emb_dim":
                EMB_DIM,

            "hidden_dim":
                HIDDEN_DIM,

            "max_len":
                MAX_LEN,

            "epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "heads":
                HEADS,

            "num_encoder_layers":
                NUM_ENCODER_LAYERS,

            "num_decoder_layers":
                NUM_DECODER_LAYERS,

            "vocab_size":
                len(token2idx),

            "dataset_size":
                len(dataset)
        }
    }


    # ========================================================
    # SAVE LATEST CHECKPOINT
    #
    # This file ALWAYS represents a completely finished epoch.
    # ========================================================

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT
    )

    print()
    print(
        "Latest checkpoint saved:"
    )

    print(
        LATEST_CHECKPOINT
    )


    # ========================================================
    # SAVE EPOCH CHECKPOINT
    # ========================================================

    epoch_checkpoint = (
        CHECKPOINT_DIR
        /
        f"epoch_{current_epoch:03d}.pt"
    )

    torch.save(
        checkpoint,
        epoch_checkpoint
    )

    print(
        f"Epoch checkpoint saved:"
    )

    print(
        epoch_checkpoint
    )


    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    if is_best:

        torch.save(
            checkpoint,
            BEST_CHECKPOINT
        )

        print()
        print(
            "Best checkpoint saved:"
        )

        print(
            BEST_CHECKPOINT
        )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    with open(
        HISTORY_FILE,
        "w"
    ) as f:

        json.dump(
            training_history,
            f,
            indent=4
        )


    # ========================================================
    # GPU MEMORY INFORMATION
    # ========================================================

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print()
        print(
            f"GPU memory allocated: "
            f"{allocated:.2f} GB"
        )

        print(
            f"GPU memory reserved: "
            f"{reserved:.2f} GB"
        )


    # ========================================================
    # EPOCH TIME
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Elapsed training time: "
        f"{elapsed / 3600:.2f} hours"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

PT_DIR = ROOT / "pt"

PT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_MODEL_PATH = (
    PT_DIR
    /
    f"{FILE_NAME}_reaction_model.pt"
)

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


# ============================================================
# FINAL TRAINING HISTORY
# ============================================================

with open(
    HISTORY_FILE,
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

total_time = (
    end_time
    -
    start_time
)

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Final epoch: "
    f"{completed_epoch}"
)

print(
    f"Best loss: "
    f"{best_loss:.6f}"
)

print(
    f"Total training time: "
    f"{total_time / 3600:.2f} hours"
)

print()
print(
    f"Final model:"
)

print(
    FINAL_MODEL_PATH
)

print()
print(
    f"Latest checkpoint:"
)

print(
    LATEST_CHECKPOINT
)

print()
print(
    f"Best checkpoint:"
)

print(
    BEST_CHECKPOINT
)

print()
print(
    f"Training history:"
)

print(
    HISTORY_FILE
)

print("=" * 70)
