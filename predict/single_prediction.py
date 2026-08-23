import time
import torch
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import warnings
from mod.model import Seq2SeqTransformer
from helper.utils import tokenize_smiles
from helper.utils import decode_indices, valid_smiles_or_empty
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
    

MODEL_PATH = ROOT / "pt" / "reaction_model.pt"
TOKEN2IDX_PATH = ROOT / "tokens" / "token2idx.json"
IDX2TOKEN_PATH = ROOT / "tokens" / "idx2token.json"



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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: model file not found at {MODEL_PATH}")
    exit()




def predict_product(reactant_smiles, max_len=120, target_smiles=None):
    model.eval()

    if not isinstance(reactant_smiles, str) or not reactant_smiles.strip():
        return {"product": "", "data": {"source_tokens": [], "target_tokens": []}}

    tokens = tokenize_smiles(reactant_smiles)
    src_ids = [token2idx.get(tok, token2idx["<unk>"]) for tok in tokens]
    if not src_ids:
        return {"product": "", "data": {"source_tokens": [], "target_tokens": []}}

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)


    with torch.no_grad():
        beam_candidates = model.beam_search_candidates(
            src_tensor, sos_idx, eos_idx, beam_width=5, max_len=max_len
        )

    for rank, (seq, score) in enumerate(beam_candidates[:5], 1):
        tokens = decode_indices(seq.tolist(), idx2token, sos_idx, eos_idx, pad_idx)
        smiles = "".join(tokens)
        
        if target_smiles is not None:
            target_canon = valid_smiles_or_empty(target_smiles)
            pred_canon = valid_smiles_or_empty(smiles)
            if target_canon and pred_canon:
                is_correct = target_canon == pred_canon
                print(f"{rank}. Score: {score:.4f} | Correct: {is_correct} | SMILES: {smiles}")
            else:
                print(f"{rank}. Score: {score:.4f} | Correct: N/A (invalid SMILES)")
        else:
            print(f"{rank}. Score: {score:.4f} | SMILES: {smiles}")
            return valid_smiles_or_empty(smiles)

    return ""


def predict_product_greedy(reactant_smiles, max_len=120):

    model.eval()

    tokens = tokenize_smiles(reactant_smiles)

    src_ids = [
        token2idx.get(tok, token2idx["<unk>"])
        for tok in tokens
    ]

    if not src_ids:
        return ""

    src_tensor = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        generated = model.greedy_decode(
            src_tensor,
            sos_idx,
            eos_idx,
            max_len=max_len
        )

    tokens = decode_indices(
        generated.squeeze(0).tolist(),
        idx2token,
        sos_idx,
        eos_idx,
        pad_idx
    )

    smiles = "".join(tokens)

    return valid_smiles_or_empty(smiles)


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
        raise ValueError(
            f"CSV must contain 'reactants' and 'products' columns: {csv_path}"
        )

    if limit is not None:
        df = df.head(limit)

    correct = 0
    checked = 0

    for _, row in df.iterrows():
        reactant = str(row["reactants"]).strip()
        target = str(row["products"]).strip()
        if not reactant or not target:
            continue

        pred = predict_product_greedy(reactant)
        pred_canon = _canonical_smiles(pred)
        target_canon = _canonical_smiles(target)

        checked += 1
        if pred_canon == target_canon:
            correct += 1

        #show progress
        if checked % 50 == 0:
            print(f"Checked: {checked}, Correct: {correct}, Accuracy: {correct/checked:.4f}")  
        
    accuracy_pct = (correct / checked * 100.0) if checked else 0.0
    return {
        "total": checked,
        "correct": correct,
        "accuracy_percent": accuracy_pct,
    }


# === Example ===
if __name__ == "__main__":
    print("Starting prediction accuracy test...")
    start_time = time.time()
    #reactant = "Brc1ccc(Br)nc1.CN(C)C=O"
    # reactant = "CCO"
    # reactant = "c1ccccc1.CC"
    # target = "CCC(O)c1ccc2c(c1)NC(=O)C(C)O2"
    reactant = "Brc1cncc(Br)c1.C[O-]"
    predicted = predict_product(reactant)
    print(f"Reactant:  {reactant}")
    print(f"Predicted: {predicted}")
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
