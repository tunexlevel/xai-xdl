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
from helper.utils import decode_indices, valid_smiles_or_empty, strip_atom_mapping
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

FILE_NAME = "uspto50k_unmapped"
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

    for seq, score in beam_candidates:
        tokens = decode_indices(seq.tolist(), idx2token, sos_idx, eos_idx, pad_idx)
        smiles = "".join(tokens)

        valid_smiles = valid_smiles_or_empty(smiles)

        if valid_smiles:
            return valid_smiles

    return ""


def predict_product_greedy(reactant_smiles, max_len=120):

    model.eval()

    tokens = tokenize_smiles(reactant_smiles)

    src_ids = [token2idx.get(tok, token2idx["<unk>"]) for tok in tokens]

    if not src_ids:
        return ""

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        generated = model.greedy_decode(src_tensor, sos_idx, eos_idx, max_len=max_len)

    tokens = decode_indices(
        generated.squeeze(0).tolist(), idx2token, sos_idx, eos_idx, pad_idx
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
    out_file = f"data/{FILE_NAME}_predicted_result.csv"

    # --------------------------------------------------------
    # Load CSV & Validate Columns
    # --------------------------------------------------------
    df = pd.read_csv(csv_path)
    if "reactants" not in df.columns or "products" not in df.columns:
        raise ValueError(
            f"CSV must contain 'reactants' and 'products' columns: {csv_path}"
        )

    if limit is not None:
        df = df.head(limit)

    valid_smiles_count = 0
    invalid_smiles_count = 0
    checked = 0
    correct = 0
    result_rows = []

    # --------------------------------------------------------
    # Evaluation Loop
    # --------------------------------------------------------
    for _, row in df.iterrows():
        reactant = str(row["reactants"]).strip()
        target = str(row["products"]).strip()

        if (
            not reactant
            or not target
            or reactant.lower() == "nan"
            or target.lower() == "nan"
        ):
            continue

        target_canon = _canonical_smiles(target)
        reactant_canon = _canonical_smiles(reactant)

        # Skip rows where dataset itself contains invalid SMILES
        if target_canon is None or reactant_canon is None:
            continue

        # Run model prediction
        pred = predict_product(reactant)
        pred_canon = _canonical_smiles(pred) if pred else None

        checked += 1

        # Check validity and canonical match
        is_valid_pred = pred_canon is not None
        is_correct = is_valid_pred and (pred_canon == target_canon)

        if is_valid_pred:
            valid_smiles_count += 1
        else:
            invalid_smiles_count += 1

        if is_correct:
            correct += 1

        # Record individual row result
        result_rows.append(
            {
                "reactant": strip_atom_mapping(reactant_canon),
                "target": strip_atom_mapping(target_canon),
                "predicted": strip_atom_mapping(pred_canon),
                "is_valid_smiles": is_valid_pred,
                "is_correct": is_correct,
            }
        )

        if checked % 200 == 0:
            accuracy = correct / checked if checked else 0.0
            valid_p = valid_smiles_count / checked if checked else 0.0
            invalid_p = invalid_smiles_count / checked if checked else 0.0

            print(
                f"Checked: {checked}, "
                f"Correct: {correct}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Valid_Smiles: {valid_smiles_count} ({valid_p:.2%}), "
                f"Invalid_Smiles: {invalid_smiles_count} ({invalid_p:.2%})"
            )

    # --------------------------------------------------------
    # Final Metrics Computation
    # --------------------------------------------------------
    accuracy_pct = (correct / checked * 100.0) if checked else 0.0
    valid_smiles_pct = (valid_smiles_count / checked * 100.0) if checked else 0.0
    invalid_smiles_pct = (invalid_smiles_count / checked * 100.0) if checked else 0.0

    print(
        f"\nFinal Summary: Checked: {checked} | Correct: {correct} | "
        f"Accuracy: {accuracy_pct:.2f}% | Valid SMILES: {valid_smiles_pct:.2f}% | "
        f"Invalid SMILES: {invalid_smiles_pct:.2f}%"
    )

    # --------------------------------------------------------
    # Build DataFrame and Append Summary Row
    # --------------------------------------------------------
    results_df = pd.DataFrame(result_rows)

    summary_row = pd.DataFrame(
        [
            {
                "reactant": f"Total Checked: {checked} ",
                "target": f"Correct: {correct}",
                "predicted": f"Accuracy: {accuracy_pct:.2f}%",
                "is_valid_smiles": f"Valid SMILES: {valid_smiles_count} ({valid_smiles_pct:.2f}%)",
                "is_correct": f"Invalid SMILES: {invalid_smiles_count} ({invalid_smiles_pct:.2f}%)",
            }
        ]
    )

    final_df = pd.concat([results_df, summary_row], ignore_index=True)

    # --------------------------------------------------------
    # Export to CSV
    # --------------------------------------------------------
    final_df.to_csv(out_file, index=False)
    print(f"Predictions successfully saved to: {out_file}")

    return {
        "total": checked,
        "correct": correct,
        "accuracy_percent": accuracy_pct,
        "valid_smiles": valid_smiles_count,
        "valid_smiles_percent": valid_smiles_pct,
        "invalid_smiles": invalid_smiles_count,
        "invalid_smiles_percent": invalid_smiles_pct,
    }
    
if __name__ == "__main__":
    print("Starting prediction accuracy test...")
    start_time = time.time()
    input_file = f"data/{FILE_NAME}_test.csv"
    
    print("" + "=" * len(input_file))
    print(input_file)
    print("" + "=" * len(input_file))
    metrics = test_prediction_accuracy(input_file)
    print(
        f"Total checked: {metrics['total']}, Correct: {metrics['correct']}, Accuracy: {metrics['accuracy_percent']:.2f} Invalid SMILES: {metrics['invalid_smiles']}, Invalid SMILES %: {metrics['invalid_smiles_percent']:.2f}, Valid SMILES: {metrics['valid_smiles']}, Valid SMILES %: {metrics['valid_smiles_percent']:.2f}"
    )
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
