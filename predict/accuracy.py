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

from helper.utils import (
    tokenize_smiles,
    decode_indices,
    valid_smiles_or_empty,
)

from rdkit import Chem, RDLogger

import pandas as pd


# ============================================================
# WARNINGS / RDKit
# ============================================================

warnings.filterwarnings(
    "ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


# ============================================================
# DEVICE / PATHS
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = ROOT / "pt" / "reaction_model.pt"

TOKEN2IDX_PATH = ROOT / "tokens" / "token2idx.json"

IDX2TOKEN_PATH = ROOT / "tokens" / "idx2token.json"


# ============================================================
# LOAD VOCABULARY
# ============================================================

try:

    with open(TOKEN2IDX_PATH, "r") as f:
        token2idx = json.load(f)

    with open(IDX2TOKEN_PATH, "r") as f:
        idx2token = {
            int(k): v
            for k, v in json.load(f).items()
        }

except FileNotFoundError:

    print(
        "❌ Error: Vocabulary files not found."
    )

    sys.exit(1)


# ============================================================
# SPECIAL TOKENS
# ============================================================

pad_idx = token2idx.get("<pad>", 0)

sos_idx = token2idx.get("<sos>", 1)

eos_idx = token2idx.get("<eos>", 2)


# ============================================================
# MODEL PARAMETERS
# ============================================================

EMB_DIM = 256
HIDDEN_DIM = 512
N_HEADS = 8
N_LAYERS = 3


# ============================================================
# LOAD MODEL
# ============================================================

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

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print("✅ Model loaded successfully.")

except FileNotFoundError:

    print(
        f"❌ Error: model file not found at {MODEL_PATH}"
    )

    sys.exit(1)


# ============================================================
# CANONICAL SMILES
# ============================================================

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

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:

        return s


# ============================================================
# TOP-K BEAM SEARCH PREDICTION
# ============================================================

def predict_top_k(
    reactant_smiles,
    k=5,
    max_len=120,
    beam_width=10
):
    """
    Generate the top K valid and unique product predictions.

    Returns a list:

        [
            {
                "product": "...",
                "canonical": "...",
                "score": ...
            },
            ...
        ]
    """

    model.eval()

    if (
        not isinstance(reactant_smiles, str)
        or not reactant_smiles.strip()
    ):
        return []

    # --------------------------------------------------------
    # Tokenize reactants
    # --------------------------------------------------------

    tokens = tokenize_smiles(
        reactant_smiles
    )

    # --------------------------------------------------------
    # Convert tokens to IDs
    # --------------------------------------------------------

    src_ids = [

        token2idx.get(
            tok,
            token2idx["<unk>"]
        )

        for tok in tokens

    ]

    if not src_ids:
        return []

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    src_tensor = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Beam search
    # --------------------------------------------------------

    with torch.no_grad():

        beam_candidates = model.beam_search_candidates(

            src_tensor,

            sos_idx,

            eos_idx,

            beam_width=beam_width,

            max_len=max_len

        )

    # --------------------------------------------------------
    # Process candidates
    # --------------------------------------------------------

    results = []

    seen = set()

    for seq, score in beam_candidates:

        # -----------------------------------------------
        # Decode
        # -----------------------------------------------

        tokens = decode_indices(

            seq.tolist(),

            idx2token,

            sos_idx,

            eos_idx,

            pad_idx

        )

        smiles = "".join(tokens)

        # -----------------------------------------------
        # Validate
        # -----------------------------------------------

        valid_smiles = valid_smiles_or_empty(
            smiles
        )

        if not valid_smiles:
            continue

        # -----------------------------------------------
        # Canonicalize
        # -----------------------------------------------

        canonical = _canonical_smiles(
            valid_smiles
        )

        if not canonical:
            continue

        # -----------------------------------------------
        # Remove duplicate products
        # -----------------------------------------------

        if canonical in seen:
            continue

        seen.add(canonical)

        # -----------------------------------------------
        # Store
        # -----------------------------------------------

        results.append({

            "product": valid_smiles,

            "canonical": canonical,

            "score": float(score)

        })

        # -----------------------------------------------
        # Stop once we have K
        # -----------------------------------------------

        if len(results) >= k:
            break

    return results


# ============================================================
# TEST ACCURACY
# ============================================================

def test_prediction_accuracy(
    csv_path,
    limit=None,
    output_csv="accuracy_test_results.csv"
):

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    if (
        "reactants" not in df.columns
        or "products" not in df.columns
    ):

        raise ValueError(

            "CSV must contain "
            "'reactants' and 'products' columns: "
            f"{csv_path}"

        )

    # --------------------------------------------------------
    # Optional limit
    # --------------------------------------------------------

    if limit is not None:

        df = df.head(limit)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    checked = 0

    top1_correct = 0

    top3_correct = 0

    top5_correct = 0

    valid_prediction_count = 0

    result_rows = []

    # ========================================================
    # LOOP
    # ========================================================

    for index, row in df.iterrows():

        reactant = str(
            row["reactants"]
        ).strip()

        target = str(
            row["products"]
        ).strip()

        if not reactant or not target:
            continue

        checked += 1

        # ----------------------------------------------------
        # Target canonical SMILES
        # ----------------------------------------------------

        target_canon = _canonical_smiles(
            target
        )

        # ----------------------------------------------------
        # Predict Top 5
        # ----------------------------------------------------

        try:

            predictions = predict_top_k(

                reactant_smiles=reactant,

                k=5,

                max_len=120,

                beam_width=10

            )

        except Exception as e:

            print(
                f"\n❌ Error on reaction {checked}: {e}"
            )

            result_rows.append({

                "reaction_index": index,

                "reactants": reactant,

                "target": target,

                "top1": "",

                "top2": "",

                "top3": "",

                "top4": "",

                "top5": "",

                "top1_correct": False,

                "top3_correct": False,

                "top5_correct": False,

                "error": str(e)

            })

            continue

        # ----------------------------------------------------
        # Valid prediction?
        # ----------------------------------------------------

        if predictions:

            valid_prediction_count += 1

        # ----------------------------------------------------
        # Canonical candidate list
        # ----------------------------------------------------

        candidate_canonicals = [

            prediction["canonical"]

            for prediction in predictions

        ]

        # ----------------------------------------------------
        # TOP 1
        # ----------------------------------------------------

        top1_match = (

            len(candidate_canonicals) >= 1

            and target_canon != ""

            and candidate_canonicals[0]
            == target_canon

        )

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        top3_match = (

            target_canon != ""

            and target_canon
            in candidate_canonicals[:3]

        )

        # ----------------------------------------------------
        # TOP 5
        # ----------------------------------------------------

        top5_match = (

            target_canon != ""

            and target_canon
            in candidate_canonicals[:5]

        )

        # ----------------------------------------------------
        # Increment metrics
        # ----------------------------------------------------

        if top1_match:

            top1_correct += 1

        if top3_match:

            top3_correct += 1

        if top5_match:

            top5_correct += 1

        
        # ====================================================
        # SAVE ROW
        # ====================================================

        result_rows.append({

            "reaction_index": index,

            "reactants": reactant,

            "target": target,

            "target_canonical": target_canon,

            "top1": (
                predictions[0]["product"]
                if len(predictions) > 0
                else ""
            ),

            "top2": (
                predictions[1]["product"]
                if len(predictions) > 1
                else ""
            ),

            "top3": (
                predictions[2]["product"]
                if len(predictions) > 2
                else ""
            ),

            "top4": (
                predictions[3]["product"]
                if len(predictions) > 3
                else ""
            ),

            "top5": (
                predictions[4]["product"]
                if len(predictions) > 4
                else ""
            ),

            "top1_correct": top1_match,

            "top3_correct": top3_match,

            "top5_correct": top5_match,

            "error": ""

        })

        # ====================================================
        # PROGRESS
        # ====================================================

        if checked % 50 == 0:

            print(
                "\n"
                f"Progress: {checked} reactions | "

                f"Top-1: "
                f"{top1_correct / checked:.4f} | "

                f"Top-3: "
                f"{top3_correct / checked:.4f} | "

                f"Top-5: "
                f"{top5_correct / checked:.4f}"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    if checked == 0:

        return {

            "total": 0,

            "top1_correct": 0,

            "top3_correct": 0,

            "top5_correct": 0,

            "top1_accuracy_percent": 0.0,

            "top3_accuracy_percent": 0.0,

            "top5_accuracy_percent": 0.0,

            "valid_prediction_percent": 0.0

        }

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    top1_accuracy = (
        top1_correct / checked * 100
    )

    top3_accuracy = (
        top3_correct / checked * 100
    )

    top5_accuracy = (
        top5_correct / checked * 100
    )

    valid_prediction_percent = (
        valid_prediction_count
        / checked
        * 100
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n\n")

    print("=" * 75)

    print("FINAL REACTION PREDICTION ACCURACY")

    print("=" * 75)

    print(
        f"Total reactions tested : {checked}"
    )

    print(
        f"Valid predictions      : "
        f"{valid_prediction_count}/{checked} "
        f"({valid_prediction_percent:.2f}%)"
    )

    print()

    print(
        f"Top-1 Accuracy         : "
        f"{top1_correct}/{checked} "
        f"({top1_accuracy:.2f}%)"
    )

    print(
        f"Top-3 Accuracy         : "
        f"{top3_correct}/{checked} "
        f"({top3_accuracy:.2f}%)"
    )

    print(
        f"Top-5 Accuracy         : "
        f"{top5_correct}/{checked} "
        f"({top5_accuracy:.2f}%)"
    )

    print("=" * 75)


    # ========================================================
    # RETURN METRICS
    # ========================================================

    return {

        "total": checked,

        "top1_correct": top1_correct,

        "top3_correct": top3_correct,

        "top5_correct": top5_correct,

        "top1_accuracy_percent":
            top1_accuracy,

        "top3_accuracy_percent":
            top3_accuracy,

        "top5_accuracy_percent":
            top5_accuracy,

        "valid_prediction_percent":
            valid_prediction_percent

    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Starting prediction accuracy test..."
    )

    start_time = time.time()

    metrics = test_prediction_accuracy(

        csv_path="data/test.csv"

    )

    print(
        "\n"
        f"Total checked: "
        f"{metrics['total']}"
    )

    print(
        f"Top-1 Accuracy: "
        f"{metrics['top1_accuracy_percent']:.2f}%"
    )

    print(
        f"Top-3 Accuracy: "
        f"{metrics['top3_accuracy_percent']:.2f}%"
    )

    print(
        f"Top-5 Accuracy: "
        f"{metrics['top5_accuracy_percent']:.2f}%"
    )

    end_time = time.time()

    print(
        f"\nTest completed in "
        f"{end_time - start_time:.2f} seconds."
    )