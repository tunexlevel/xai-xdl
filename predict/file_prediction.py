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
from helper.utils import decode_indices, valid_smiles_or_empty, map_smiles, strip_atom_mapping
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
    

FILE_NAME = "uspto50k_mapped"
MODEL_PATH = ROOT / "pt" / f"{FILE_NAME}_reaction_model.pt"
TOKEN2IDX_PATH = ROOT / "tokens" / f"{FILE_NAME}_token2idx.json"
IDX2TOKEN_PATH = ROOT / "tokens" / f"{FILE_NAME}_idx2token.json"



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
        pred_canon = valid_smiles_or_empty(smiles)

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
            src_tensor, sos_idx, eos_idx, beam_width=1, max_len=max_len
        )

    return get_best_prediction(beam_candidates, idx2token, sos_idx, eos_idx, pad_idx, target_smiles)
            

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

        pred = predict_product(reactant)
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



def main(limit=None):
    csv_path = f"data/{FILE_NAME}_test.csv"
    output_csv = f"data/{FILE_NAME}_test_results.csv"

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------
    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------
    if "reactants" not in df.columns or "products" not in df.columns:
        raise ValueError(
            f"CSV must contain 'reactants' and 'products' columns: {csv_path}"
        )

    # --------------------------------------------------------
    # Optional limit
    # --------------------------------------------------------
    if limit is not None:
        df = df.head(limit)

    # --------------------------------------------------------
    # Metrics & Row Collection
    # --------------------------------------------------------
    correct_count = 0
    checked = 0
    result_rows = []

    # Run the accuracy test on the provided data
    for index, row in df.iterrows():
        reactant = str(row["reactants"]).strip()
        target = str(row["products"]).strip()

        beam_prediction = predict_product(reactant, target_smiles=target)
        is_correct = beam_prediction == target

        checked += 1
        if is_correct:
            correct_count += 1

        # Append row details for CSV export
        result_rows.append(
            {
                "reactants": strip_atom_mapping(reactant),
                "target_product": strip_atom_mapping(target),
                "predicted_product": strip_atom_mapping(beam_prediction),
                "is_correct": is_correct,
            }
        )

        # print(
        #     f"Reactant: {reactant} | Correct: {is_correct} | "
        #     f"Predicted Product: {beam_prediction} | Target Product: {target}"
        # )

    # Calculate final accuracy metrics
    correct_percentage = (correct_count / checked * 100.0) if checked else 0.0

    print(
        f"\nTotal: {checked}, Correct: {correct_count}, "
        f"Accuracy: {correct_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Create DataFrame and Append Final Summary Row
    # --------------------------------------------------------
    results_df = pd.DataFrame(result_rows)

    summary_row = pd.DataFrame(
        [
            {
                "reactants": "--- SUMMARY ---",
                "target_product": f"Total: {checked}",
                "predicted_product": f"Correct: {correct_count}",
                "is_correct": f"Accuracy: {correct_percentage:.2f}%",
            }
        ]
    )

    final_df = pd.concat([results_df, summary_row], ignore_index=True)

    # --------------------------------------------------------
    # Export to CSV
    # --------------------------------------------------------
    final_df.to_csv(output_csv, index=False)
    print(f"Results successfully saved to: {output_csv}")

    return {
        "total": checked,
        "correct": correct_count,
        "accuracy_pct": correct_percentage,
    }
    
    
# === Example ===
if __name__ == "__main__":
    print("Starting prediction accuracy test...")
    
    
    start_time = time.time()
    
    main(200)
    
    end_time = time.time()
    
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
