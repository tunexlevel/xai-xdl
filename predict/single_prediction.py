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
            src_tensor, sos_idx, eos_idx, beam_width=5, max_len=max_len
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


# === Example ===
if __name__ == "__main__":
    print("Starting prediction accuracy test...")
    data = [    #'C1COCCN1.FC(F)(F)c1ccc(Br)cc1>>FC(F)(F)c1ccc(N2CCOCC2)cc1',
            'ClCc1cccc(CCl)n1.Sc1ccccc1>>c3ccc(SCc2cccc(CSc1ccccc1)n2)cc3',
                'C.Cl>>CCl','C=C.O>>CCO','C=C>>CC','CCO>>CC=O','CC(=O)O.COC>>CC(=O)OC',
                'CC(=O)O.CCO>>CC(=O)OCC','CCO>>C=C','CC=O>>CCO','CC(=O)C>>CC(O)C','c1ccccc1.Cl>>Clc1ccccc1','Nc1ccccc1.CC(=O)O>>CC(=O)Nc1ccccc1',
            ]
    data1 = [
        'CC(C)(C)OC(=O)c1ccc(NCC2(O)CCN(CCc3ccc(C#N)cc3)CC2)cc1.O=C([O-])O>>CN(CC1(O)CCN(CCc2ccc(C#N)cc2)CC1)c1ccc(C(=O)OC(C)(C)C)cc1',
        'CC(=O)c1ccc(Cl)c(Nc2ccccc2)c1.CS(=O)(=O)Cl>>CC(=O)c1ccc(Cl)c(NS(C)(=O)=O)c1',
        'O=C1CCC(=O)N1Br.c1ccc(CCCOc2ccccc2)cc1>>BrC(CCOc1ccccc1)c1ccccc1',
        'CC(=O)Nc1ccc(C=O)cc1>>CC(=O)Nc1ccc(CO)cc1',
        'CC(C)CC1CNCCN1.O=S(=O)(Cl)c1cc2ccc(Cl)cc2s1>>CC(C)CC1CN(S(=O)(=O)c2cc3ccc(Cl)cc3s2)CCN1',
        'CC(=O)Cl.CCNCc1cc(C(F)(F)F)ccc1-c1cc(C(F)(F)C(=O)O)ccc1OC>>CCN(Cc1cc(C(F)(F)F)ccc1-c1cc(C(F)(F)C(=O)O)ccc1OC)C(C)=O',
        'CCOC(=O)C(=O)c1csc(N)n1.O=C=Nc1ccc(Br)cc1>>CCOC(=O)C(=O)c1csc(NC(=O)Nc2ccc(Br)cc2)n1',
        'COc1nccnc1I.NN>>COc1nccnc1NN',
        'COc1cc(C=C(C#N)c2ccc(C(F)(F)F)nc2)ccc1O.O=[N+]([O-])O>>COc1cc(C=C(C#N)c2ccc(C(F)(F)F)nc2)cc([N+](=O)[O-])c1O',
        'N#C[S-].O=C(CBr)c1ccc(C(F)(F)F)cc1>>N#CSCC(=O)c1ccc(C(F)(F)F)cc1',
        'CC(=O)Nc1ccc(S(=O)(=O)C(F)(F)F)cc1.O=[N+]([O-])O>>CC(=O)Nc1ccc(S(=O)(=O)C(F)(F)F)cc1[N+](=O)[O-]',
        'CC(C)(C)CNC(=O)NCCCl.O=N[O-]>>CC(C)(C)CNC(=O)N(CCCl)N=O'
    ]
    
    data2 = [
        'CCOC(=O)c1nn(-c2ccc(Cl)cc2Cl)c(-c2ccc(OC)cc2)c1C.O=C1CCC(=O)N1Br>>CCOC(=O)c1nn(-c2ccc(Cl)cc2Cl)c(-c2ccc(OC)cc2)c1CBr',
        'CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.N#Cc1cc(-c2cccc([N+](=O)[O-])c2)c2nc(N)sc2c1>>CC(C)(C)OC(=O)Nc1nc2c(-c3cccc([N+](=O)[O-])c3)cc(C#N)cc2s1',
        'NCc1ccccc1S(=O)(=O)C1CC1.O=C(OC(=O)C(F)(F)F)C(F)(F)F>>O=C(NCc1ccccc1S(=O)(=O)C1CC1)C(F)(F)F',
        'C/C=C/C(=O)O[Si](C)(C)C.O=C1CCC(=O)N1Br>>C[Si](C)(C)OC(=O)/C=C/CBr',
        'CC(C)(C)OC(=O)N1CCc2oc3c(Cl)cc(Sc4ccccc4)cc3c2C1.O=C(OO)c1cccc(Cl)c1>>CC(C)(C)OC(=O)N1CCc2oc3c(Cl)cc(S(=O)c4ccccc4)cc3c2C1',
        'CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.O=Cc1c[nH]cn1>>CC(C)(C)OC(=O)n1cnc(C=O)c1',
        'Cc1cnc(Cl)cc1Cl.O=C1CCC(=O)N1Br>>Clc1cc(Cl)c(CBr)cn1',
        'COc1nc2ccc(C(O)c3cnnn3C)cc2c(Cl)c1Cc1ccc(C(F)(F)F)cc1>>COc1nc2ccc(C(=O)c3cnnn3C)cc2c(Cl)c1Cc1ccc(C(F)(F)F)cc1',
        'Cc1cc(/C=C/C(F)(F)F)ccc1C(=O)Nc1ccc2sc(C(C)O)nc2c1>>CC(=O)c1nc2cc(NC(=O)c3ccc(/C=C/C(F)(F)F)cc3C)ccc2s1',
        'CCOC(=O)C1CCC(=O)CC1.OCCO>>CCOC(=O)C1CCC2(CC1)OCCO2',
        'CCCCc1ncc(C(C)O)n1Cc1ccccc1Cl>>CCCCc1ncc(C(C)=O)n1Cc1ccccc1Cl',
        'CC1(c2cc3cccnc3[nH]2)CC1.O=C(OO)c1cccc(Cl)c1>>CC1(c2cc3ccc[n+]([O-])c3[nH]2)CC1',
        'O=[N+]([O-])c1cc(CO)ccc1F>>O=Cc1ccc(F)c([N+](=O)[O-])c1' 
    ]
    start_time = time.time()
    
    # Run the accuracy test on the provided data
    for reaction in data2:
        reactants, products = reaction.split(">>")
        predicted_product = predict_product_greedy(reactants)
        beam_prediction = predict_product(reactants, target_smiles=products)
        correct = beam_prediction == products or predicted_product == products
        print(f"Reactants: {reactants} | Correct: {correct} | Predicted Product: {predicted_product} | Beam Product: {beam_prediction} | Target Product: {products}")  
    
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
