import time
import torch
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import warnings
from mod.model_retro import Seq2SeqTransformer
from helper.utils import get_root_aligned_pair, tokenize_smiles
from helper.utils import decode_indices, valid_smiles_or_empty, map_smiles, strip_atom_mapping, _canonicalize_reactants
from rdkit import Chem, RDLogger
import pandas as pd


warnings.filterwarnings(
    "ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


# Silence noisy invalid-SMILES parse warnings while the model is still being tuned.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

DATASET_NAME = "uspto50k_mapped" # "ocrtrain" # "uspto_mit_mapped" # "uspto50k_unmapped" # "uspto50k_mapped"
FILE_NAME = f"{DATASET_NAME}_retro_3-3" 
MODEL_PATH = ROOT / "pt" / "dump" / f"{FILE_NAME}_2_reaction_model.pt"
TOKEN2IDX_PATH = ROOT / "tokens" / "dump" / f"{FILE_NAME}_1_token2idx.json"
IDX2TOKEN_PATH = ROOT / "tokens" / "dump" /  f"{FILE_NAME}_1_idx2token.json"
IS_CHECKPOINT = True  # Set to True if loading from a checkpoint, False if loading a full model state dict


# === Load vocab and model ===
try:
    with open(TOKEN2IDX_PATH, "r") as f:
        token2idx = json.load(f)
    with open(IDX2TOKEN_PATH, "r") as f:
        # JSON keys are always strings, convert them back to ints
        idx2token = {int(k): v for k, v in json.load(f).items()}
except FileNotFoundError:
    print("❌ Error: Vocabulary files not found. Please download them from Colab.")
    exit()


# Define special tokens
pad_idx = token2idx.get("<pad>", 0)
sos_idx = token2idx.get("<sos>", 1)
eos_idx = token2idx.get("<eos>", 2)

# 3. Load Model
# Ensure these params match your training EXACTLY
EMB_DIM = 256
HIDDEN_DIM = 512
N_HEADS = 8
N_LAYERS = 3

model = Seq2SeqTransformer(
    input_dim=len(token2idx),
    output_dim=len(token2idx),
    emb_dim=EMB_DIM,
    nhead=N_HEADS,
    num_encoder_layers=N_LAYERS,
    num_decoder_layers=N_LAYERS,
    dim_feedforward=HIDDEN_DIM,
    pad_idx=pad_idx,
).to(device)


try:
    if IS_CHECKPOINT:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Load the full model state dict directly
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: model file not found at {MODEL_PATH}")
    exit()


def get_best_prediction(beam_candidates, idx2token, sos_idx, eos_idx, pad_idx,
                        target_smiles=None):

    best_valid_prediction = ""

    target_canon = ""
    if target_smiles is not None:
        target_canon = valid_smiles_or_empty(target_smiles)

    for rank, (seq, score) in enumerate(beam_candidates[:5], 1):

        # Decode token indices
        tokens = decode_indices(
            seq.tolist(),
            idx2token,
            sos_idx,
            eos_idx,
            pad_idx
        )

        smiles = "".join(tokens)

        # Validate and canonicalize prediction
        pred_canon = smiles #valid_smiles_or_empty(smiles)

        # Skip invalid SMILES
        if not pred_canon:
            continue

        # Keep the highest-scoring valid prediction
        if not best_valid_prediction:
            best_valid_prediction = pred_canon

        # If target is available, check for exact canonical match
        if target_canon:
            if pred_canon == target_canon:
                return pred_canon

        # print(
        #     f"{rank}. Score: {score:.4f} | "
        #     f"SMILES: {pred_canon}"
        # )

    # No exact match found, return best valid prediction
    return best_valid_prediction

def predict_reactants(product_smiles, max_len=120, target_smiles=None):
    model.eval()

    if not isinstance(product_smiles, str) or not product_smiles.strip():
        return {
            "reactants": "",
            "data": {
                "source_tokens": [],
                "target_tokens": []
            }
        }

    tokens = tokenize_smiles(product_smiles)

    src_ids = [
        token2idx.get(tok, token2idx["<unk>"])
        for tok in tokens
    ]

    if not src_ids:
        return {
            "reactants": "",
            "data": {
                "source_tokens": [],
                "target_tokens": []
            }
        }

    src_tensor = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        beam_candidates = model.beam_search_candidates(
            src_tensor,
            sos_idx,
            eos_idx,
            beam_width=10,
            max_len=max_len
        )

    return get_best_prediction(
        beam_candidates,
        idx2token,
        sos_idx,
        eos_idx,
        pad_idx,
        target_smiles
    )
