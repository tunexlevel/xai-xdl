import torch
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "pt" / "reaction_model.pt"
TOKEN2IDX_PATH = ROOT / "tokens" / "token2idx.json"
IDX2TOKEN_PATH = ROOT / "tokens" / "idx2token.json"

from mod.model import Seq2SeqTransformer
from helper.utils import tokenize_smiles
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

# Silence noisy invalid-SMILES parse warnings while the model is still being tuned.
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    pad_idx=pad_idx
).to(device)


try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: model file not found at {MODEL_PATH}")
    exit()


def _decode_indices(indices):
    result = []
    for idx in indices:
        if idx == sos_idx:
            continue
        if idx == eos_idx:
            break
        if idx == pad_idx:
            continue
        token = idx2token.get(int(idx))
        if token is not None:
            result.append(token)
    return result


def _valid_smiles_or_empty(smiles):
    if not isinstance(smiles, str):
        return ""
    cleaned = smiles.strip().replace("<pad>", "")
    if not cleaned:
        return ""
    try:
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _syntax_score(smiles):
    if not smiles:
        return -1e9
    score = 0.0
    score -= abs(smiles.count('(') - smiles.count(')')) * 10.0
    score -= max(0, smiles.count(')') - smiles.count('(')) * 25.0
    score -= max(0, len(smiles) - 120) * 5.0
    return score


def predict_product(reactant_smiles, max_len=120):
    model.eval()

    if not isinstance(reactant_smiles, str) or not reactant_smiles.strip():
        return {"product": "", "data": {"source_tokens": [], "target_tokens": []}}

    tokens = tokenize_smiles(reactant_smiles)
    src_ids = [token2idx.get(tok, token2idx["<unk>"]) for tok in tokens]
    if not src_ids:
        return {"product": "", "data": {"source_tokens": [], "target_tokens": []}}

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    best_result = {"smiles": "", "score": -1e9, "tokens": []}

    with torch.no_grad():
        beam_candidates = model.beam_search_candidates(src_tensor, sos_idx, eos_idx, beam_width=12, max_len=max_len)

    for seq, score in beam_candidates:
        raw_tokens = _decode_indices(seq.squeeze(0).tolist())
        raw_smiles = "".join(raw_tokens)
        valid_smiles = _valid_smiles_or_empty(raw_smiles)
        candidate_score = _syntax_score(raw_smiles)
        if valid_smiles:
            candidate_score += 1000.0
        if candidate_score > best_result["score"]:
            best_result = {"smiles": valid_smiles if valid_smiles else raw_smiles, "score": candidate_score, "tokens": raw_tokens}

    if not best_result["smiles"]:
        generated, _ = model.generate(src_tensor, sos_idx, eos_idx)
        fallback = _decode_indices(generated.squeeze(0).tolist())
        fallback_smiles = "".join(fallback)
        best_result = {"smiles": _valid_smiles_or_empty(fallback_smiles) or fallback_smiles, "score": _syntax_score(fallback_smiles), "tokens": fallback}

    return best_result["smiles"]


def _canonical_smiles(smiles):
    if not isinstance(smiles, str):
        return ""
    s = smiles.strip()
    if not s:
        return ""
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return s
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return s


def test_prediction_accuracy(csv_path="data/uspto50k/tested.csv", limit=None):
    df = pd.read_csv(csv_path)
    if "reactants" not in df.columns or "products" not in df.columns:
        raise ValueError(f"CSV must contain 'reactants' and 'products' columns: {csv_path}")

    if limit is not None:
        df = df.head(limit)

    correct = 0
    checked = 0

    for _, row in df.iterrows():
        reactant = str(row["reactants"]).strip()
        target = str(row["products"]).strip()
        if not reactant or not target:
            continue

        pred = predict_product(reactant)
        pred_canon = _canonical_smiles(pred)
        target_canon = _canonical_smiles(target)

        checked += 1
        if pred_canon == target_canon:
            correct += 1

    accuracy_pct = (correct / checked * 100.0) if checked else 0.0
    return {
        "total": checked,
        "correct": correct,
        "accuracy_percent": accuracy_pct,
    }

# === Example ===
if __name__ == "__main__":
    # reactant = "Brc1ccc(Br)nc1.CN(C)C=O"
    # reactant = "c1ccccc1.Cl"
    # predicted = predict_product(reactant)
    # print(f"Reactant:  {reactant}")
    # print(f"Predicted: {predicted}")

    metrics = test_prediction_accuracy("data/uspto50k/tested.csv", limit=40)
    print(metrics)

    # try:
    #     mol1 = Chem.MolFromSmiles(reactant)
    #     mol2 = Chem.MolFromSmiles(predicted)
    #     Draw.MolsToGridImage([mol1, mol2], legends=["Reactant", "Predicted Product"]).show()
    # except Exception as e:
    #     print("Visualization failed:", e)
