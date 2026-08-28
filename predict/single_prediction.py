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
from helper.utils import decode_indices, valid_smiles_or_empty, map_smiles
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
            src_tensor, sos_idx, eos_idx, beam_width=10, max_len=max_len
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


def main(file_name):
    simple_ocr = [   
            'ClCc1cccc(CCl)n1.Sc1ccccc1,c3ccc(SCc2cccc(CSc1ccccc1)n2)cc3',
            'C.Cl,CCl','C=C.O,CCO','C=C,CC','CCO,CC=O','CC(=O)O.COC,CC(=O)OC',
            'CC(=O)O.CCO,CC(=O)OCC','CCO,C=C','CC=O,CCO','CC(=O)C,CC(O)C','c1ccccc1.Cl,Clc1ccccc1','Nc1ccccc1.CC(=O)O,CC(=O)Nc1ccccc1',
        ]
    
    # USPTO-50k test set (unmapped)
    upsto_unmapped_test = [
            'CCOC(=O)c1nn(-c2ccc(Cl)cc2Cl)c(-c2ccc(OC)cc2)c1C.O=C1CCC(=O)N1Br,CCOC(=O)c1nn(-c2ccc(Cl)cc2Cl)c(-c2ccc(OC)cc2)c1CBr',
            'CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.N#Cc1cc(-c2cccc([N+](=O)[O-])c2)c2nc(N)sc2c1,CC(C)(C)OC(=O)Nc1nc2c(-c3cccc([N+](=O)[O-])c3)cc(C#N)cc2s1',
            'NCc1ccccc1S(=O)(=O)C1CC1.O=C(OC(=O)C(F)(F)F)C(F)(F)F,O=C(NCc1ccccc1S(=O)(=O)C1CC1)C(F)(F)F',
            'C/C=C/C(=O)O[Si](C)(C)C.O=C1CCC(=O)N1Br,C[Si](C)(C)OC(=O)/C=C/CBr',
            'CC(C)(C)OC(=O)N1CCc2oc3c(Cl)cc(Sc4ccccc4)cc3c2C1.O=C(OO)c1cccc(Cl)c1,CC(C)(C)OC(=O)N1CCc2oc3c(Cl)cc(S(=O)c4ccccc4)cc3c2C1',
            'CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.O=Cc1c[nH]cn1,CC(C)(C)OC(=O)n1cnc(C=O)c1',
            'Cc1cnc(Cl)cc1Cl.O=C1CCC(=O)N1Br,Clc1cc(Cl)c(CBr)cn1',
            'COc1nc2ccc(C(O)c3cnnn3C)cc2c(Cl)c1Cc1ccc(C(F)(F)F)cc1,COc1nc2ccc(C(=O)c3cnnn3C)cc2c(Cl)c1Cc1ccc(C(F)(F)F)cc1',
            'Cc1cc(/C=C/C(F)(F)F)ccc1C(=O)Nc1ccc2sc(C(C)O)nc2c1,CC(=O)c1nc2cc(NC(=O)c3ccc(/C=C/C(F)(F)F)cc3C)ccc2s1',
            'CCOC(=O)C1CCC(=O)CC1.OCCO,CCOC(=O)C1CCC2(CC1)OCCO2',
        ]

    # USPTO-50k test set (mapped)
    upsto_mapped_test = [
            'O=C1CCC(=O)N1[Br:1].[CH3:2][CH2:3][O:4][C:5](=[O:6])[c:7]1[n:8][n:9](-[c:10]2[cH:11][cH:12][c:13]([Cl:14])[cH:15][c:16]2[Cl:17])[c:18](-[c:19]2[cH:20][cH:21][c:22]([O:23][CH3:24])[cH:25][cH:26]2)[c:27]1[CH3:28],[Br:1][CH2:28][c:27]1[c:7]([C:5]([O:4][CH2:3][CH3:2])=[O:6])[n:8][n:9](-[c:10]2[cH:11][cH:12][c:13]([Cl:14])[cH:15][c:16]2[Cl:17])[c:18]1-[c:19]1[cH:20][cH:21][c:22]([O:23][CH3:24])[cH:25][cH:26]1',
            'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[N:8]#[C:9][c:10]1[cH:11][c:12](-[c:13]2[cH:14][cH:15][cH:16][c:17]([N+:18](=[O:19])[O-:20])[cH:21]2)[c:22]2[n:23][c:24]([NH2:25])[s:26][c:27]2[cH:28]1,[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[NH:25][c:24]1[n:23][c:22]2[c:12](-[c:13]3[cH:14][cH:15][cH:16][c:17]([N+:18](=[O:19])[O-:20])[cH:21]3)[cH:11][c:10]([C:9]#[N:8])[cH:28][c:27]2[s:26]1',
            'O=C(O[C:1](=[O:2])[C:3]([F:4])([F:5])[F:6])C(F)(F)F.[NH2:7][CH2:8][c:9]1[cH:10][cH:11][cH:12][cH:13][c:14]1[S:15](=[O:16])(=[O:17])[CH:18]1[CH2:19][CH2:20]1,[C:1](=[O:2])([C:3]([F:4])([F:5])[F:6])[NH:7][CH2:8][c:9]1[cH:10][cH:11][cH:12][cH:13][c:14]1[S:15](=[O:16])(=[O:17])[CH:18]1[CH2:19][CH2:20]1',
            'O=C1CCC(=O)N1[Br:1].[CH3:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11],[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]',
            'O=C(O[OH:1])c1cccc(Cl)c1.[CH3:2][C:3]([CH3:4])([CH3:5])[O:6][C:7](=[O:8])[N:9]1[CH2:10][CH2:11][c:12]2[o:13][c:14]3[c:15]([Cl:16])[cH:17][c:18]([S:19][c:20]4[cH:21][cH:22][cH:23][cH:24][cH:25]4)[cH:26][c:27]3[c:28]2[CH2:29]1,[O:1]=[S:19]([c:18]1[cH:17][c:15]([Cl:16])[c:14]2[o:13][c:12]3[c:28]([c:27]2[cH:26]1)[CH2:29][N:9]([C:7]([O:6][C:3]([CH3:2])([CH3:4])[CH3:5])=[O:8])[CH2:10][CH2:11]3)[c:20]1[cH:21][cH:22][cH:23][cH:24][cH:25]1',
            'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[O:8]=[CH:9][c:10]1[cH:11][nH:12][cH:13][n:14]1,[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:12]1[cH:11][c:10]([CH:9]=[O:8])[n:14][cH:13]1',
            'O=C1CCC(=O)N1[Br:1].[CH3:2][c:3]1[cH:4][n:5][c:6]([Cl:7])[cH:8][c:9]1[Cl:10],[Br:1][CH2:2][c:3]1[cH:4][n:5][c:6]([Cl:7])[cH:8][c:9]1[Cl:10]',
            '[CH3:1][O:2][c:3]1[n:4][c:5]2[cH:6][cH:7][c:8]([CH:9]([OH:10])[c:11]3[cH:12][n:13][n:14][n:15]3[CH3:16])[cH:17][c:18]2[c:19]([Cl:20])[c:21]1[CH2:22][c:23]1[cH:24][cH:25][c:26]([C:27]([F:28])([F:29])[F:30])[cH:31][cH:32]1,[CH3:1][O:2][c:3]1[n:4][c:5]2[cH:6][cH:7][c:8]([C:9](=[O:10])[c:11]3[cH:12][n:13][n:14][n:15]3[CH3:16])[cH:17][c:18]2[c:19]([Cl:20])[c:21]1[CH2:22][c:23]1[cH:24][cH:25][c:26]([C:27]([F:28])([F:29])[F:30])[cH:31][cH:32]1',
            '[CH3:1][c:2]1[cH:3][c:4](/[CH:5]=[CH:6]/[C:7]([F:8])([F:9])[F:10])[cH:11][cH:12][c:13]1[C:14](=[O:15])[NH:16][c:17]1[cH:18][cH:19][c:20]2[s:21][c:22]([CH:23]([CH3:24])[OH:25])[n:26][c:27]2[cH:28]1,[CH3:1][c:2]1[cH:3][c:4](/[CH:5]=[CH:6]/[C:7]([F:8])([F:9])[F:10])[cH:11][cH:12][c:13]1[C:14](=[O:15])[NH:16][c:17]1[cH:18][cH:19][c:20]2[s:21][c:22]([C:23]([CH3:24])=[O:25])[n:26][c:27]2[cH:28]1',
            'O[CH2:1][CH2:2][OH:3].[CH3:4][CH2:5][O:6][C:7](=[O:8])[CH:9]1[CH2:10][CH2:11][C:12](=[O:13])[CH2:14][CH2:15]1,[CH2:1]1[CH2:2][O:3][C:12]2([CH2:11][CH2:10][CH:9]([C:7]([O:6][CH2:5][CH3:4])=[O:8])[CH2:15][CH2:14]2)[O:13]1',
        ]
    
    correct_count = 0
    checked = 0
    correct_percentage = 0
    
    if(file_name == 'uspto50k_unmapped'):
        data_set = upsto_unmapped_test
        
    if(file_name == 'uspto50k_mapped'):
            data_set = upsto_mapped_test
        
    
    # Run the accuracy test on the provided data
    for reaction in data_set:
        reactants, products = reaction.split(",")
        # reactants = map_smiles(reactants)
        # products = map_smiles(products)
        predicted_product = predict_product_greedy(reactants)
        beam_prediction = predict_product(reactants, target_smiles=products)
        correct = beam_prediction == products or predicted_product == products
        checked += 1
        if correct:
            correct_count += 1
        print(f"Reactants: {reactants} | Correct: {correct} | Predicted Product: {predicted_product} | Beam Product: {beam_prediction} | Target Product: {products}")  
    
    correct_percentage = correct_count/checked
    
    print(f"Total: {checked}, Correct: {correct_count}, Accuracy: {correct_percentage}")

# === Example ===
if __name__ == "__main__":
    print("Starting prediction accuracy test...")
    
    
    start_time = time.time()
    
    main(FILE_NAME)
    
    end_time = time.time()
    
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
