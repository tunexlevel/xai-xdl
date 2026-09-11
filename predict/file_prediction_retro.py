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
    

DATASET_NAME = "uspto50k_unmapped" # "ocrtrain" # "uspto_mit_mapped" # "uspto50k_unmapped" # "uspto50k_mapped"
FILE_NAME = f"{DATASET_NAME}_retro_3-3" 
MODEL_PATH = ROOT / "pt" /"dump" /  f"{FILE_NAME}_reaction_model.pt"
TOKEN2IDX_PATH = ROOT / "tokens" / "dump" /  f"{FILE_NAME}_token2idx.json"
IDX2TOKEN_PATH = ROOT / "tokens" / "dump" /  f"{FILE_NAME}_idx2token.json"



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


def get_best_prediction(
    beam_candidates,
    idx2token,
    sos_idx,
    eos_idx,
    pad_idx
):
    """
    Return the highest-scoring valid beam prediction.

    IMPORTANT:
    The target reaction must NOT be supplied here.
    """

    for rank, (seq, score) in enumerate(beam_candidates, 1):

        # Decode token IDs
        tokens = decode_indices(
            seq.tolist(),
            idx2token,
            sos_idx,
            eos_idx,
            pad_idx
        )

        smiles = "".join(tokens)

        # Validate + canonicalize
        pred_canon = valid_smiles_or_empty(smiles)

        # Skip invalid predictions
        if not pred_canon:
            continue

        # First valid candidate is the highest-scoring
        # valid beam prediction
        return pred_canon

    # No valid prediction
    return ""

def predict_reactants(product_smiles, max_len=160):
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
            beam_width=5,
            max_len=max_len
        )

    return get_best_prediction(
        beam_candidates,
        idx2token,
        sos_idx,
        eos_idx,
        pad_idx,
    )
         
         
def predict_reactants_greedy(product_smiles, max_len=160):
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

    # return valid_smiles_or_empty(smiles)
    return smiles

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



def _canonicalize_reactants(smiles):
    mols = []

    for component in smiles.split("."):
        mol = Chem.MolFromSmiles(component.strip())

        if mol is None:
            return None

        mols.append(Chem.MolToSmiles(mol, canonical=True))

    return ".".join(sorted(mols))

def test_prediction_accuracy(csv_path="data/test.csv", limit=None):
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
        product = str(row["products"]).strip()
        if not reactant or not product:
            continue

        pred = predict_reactants_greedy(product, max_len=160, target_smiles=reactant)
        pred_canon = _canonicalize_reactants(pred)
        target_canon = _canonicalize_reactants(reactant)

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
    # csv_path = f"data/{DATASET_NAME}_test.csv"
    csv_path = f"data/test.csv"
    output_csv = f"data/{DATASET_NAME}_test_results.csv"

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
    greedy_correct_count = 0
    beam_correct_count = 0
    checked = 0
    result_rows = []

    # Run the accuracy test on the provided data
    for index, row in df.iterrows():
        reactants = str(row["reactants"]).strip()
        products = str(row["products"]).strip()

        beam_prediction = predict_reactants(
            product_smiles=products,
        )
        
        greedy_prediction = predict_reactants_greedy(
            product_smiles=products,
        )
        
        beam_canon = _canonical_smiles(beam_prediction)
        greedy_canon = _canonical_smiles(greedy_prediction)
        target_canon = _canonical_smiles(reactants)

        is_beam_correct = (
            bool(beam_canon) and
            beam_canon == target_canon
        )

        is_greedy_correct = (
            bool(greedy_canon) and
            greedy_canon == target_canon
        )

        checked += 1
        if is_beam_correct:
            beam_correct_count += 1
        
        if is_greedy_correct:
            greedy_correct_count += 1

        # Append row details for CSV export
        result_rows.append(
            {
                "reactants": strip_atom_mapping(reactants),
                "products": strip_atom_mapping(products),
                "beam_predicted_reactants": strip_atom_mapping(beam_prediction),
                "greedy_predicted_reactants": strip_atom_mapping(greedy_prediction),
                "is_beam_correct": is_beam_correct,
                "is_greedy_correct": is_greedy_correct,
            }
        )


    # Calculate final accuracy metrics
    beam_accuracy_pct = (beam_correct_count / checked * 100.0) if checked else 0.0
    greedy_accuracy_pct = (greedy_correct_count / checked * 100.0) if checked else 0.0

    print(
        f"\nTotal: {checked}, Beam Accuracy: {beam_accuracy_pct:.2f}%, "
        f"Greedy Accuracy: {greedy_accuracy_pct:.2f}%"
    )

    # --------------------------------------------------------
    # Create DataFrame and Append Final Summary Row
    # --------------------------------------------------------
    results_df = pd.DataFrame(result_rows)

    summary_row = pd.DataFrame(
        [
            {
                "reactants": "--- SUMMARY ---",
                "products": f"Total: {checked}",
                "beam_predicted_reactants": f"Correct: {beam_correct_count} (Beam)",
                "greedy_predicted_reactants": f"Correct: {greedy_correct_count} (Greedy)",
                "is_beam_correct": f"Accuracy: {beam_accuracy_pct:.2f}%",
                "is_greedy_correct": f"Accuracy: {greedy_accuracy_pct:.2f}%",
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
        "beam_correct": beam_correct_count,
        "greedy_correct": greedy_correct_count,
        "beam_accuracy_pct": beam_accuracy_pct,
        "greedy_accuracy_pct": greedy_accuracy_pct,
    }
    
    
# === Example ===
if __name__ == "__main__":
    print("Starting prediction accuracy test...")
    
    
    start_time = time.time()
    
    main(200)
    
    end_time = time.time()
    
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
