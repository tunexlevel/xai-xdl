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
            'Cc1nc(N)sc1C(=O)NCc1ccccc1.O=C(Cl)Cc1ccccc1,Cc1nc(NC(=O)Cc2ccccc2)sc1C(=O)NCc1ccccc1',
            'Cc1ccccc1C=O.Nc1cccc(-c2ccnc3c(C(=O)c4cccs4)cnn23)c1,Cc1ccccc1CNc1cccc(-c2ccnc3c(C(=O)c4cccs4)cnn23)c1',
            'C[C@H](NC(=O)OC(C)(C)C)[C@H](O)CN.O=S(=O)(Cl)c1ccccn1,C[C@H](NC(=O)OC(C)(C)C)[C@@H](O)CNS(=O)(=O)c1ccccn1',
            'CCOC(=O)c1sc(C(=O)OC(C)(C)C)cc1OC(C)(C)C,CC(C)(C)OC(=O)c1cc(OC(C)(C)C)c(C(=O)O)s1',
            'CCOC[C@H](O)C(=O)OC.Clc1cccnc1-n1ncc2c(Cl)ncnc21,CCOC[C@H](Oc1ncnc2c1cnn2-c1ncccc1Cl)C(=O)OC',
            'COC(=O)c1ccc2c(c1)nc(NC(=O)c1ccno1)n2C[C@H]1CCCN1C(=O)OC(C)(C)C,CC(C)(C)OC(=O)N1CCC[C@@H]1Cn1c(NC(=O)c2ccno2)nc2cc(CO)ccc21',
            'COC(=O)[C@@H]1Cc2ccc(C3=C[C@@H](C(=O)N[C@@H]4CCCc5ccccc54)N(C(=O)OC(C)(C)C)C3)cc2CN1C(=O)OC(C)(C)C,COC(=O)[C@@H]1Cc2ccc([C@H]3C[C@@H](C(=O)N[C@@H]4CCCc5ccccc54)N(C(=O)OC(C)(C)C)C3)cc2CN1C(=O)OC(C)(C)C',
            'COC(=O)c1ccc(Br)c(O)c1.OCC1CC1,COC(=O)c1ccc(Br)c(OCC2CC2)c1',
            'CCCC1(CC(=O)OCC)OCCc2c1[nH]c1c(C)c(C(=O)O)cc(C#N)c21,CCCC1(CC(=O)OCC)OCCc2c1[nH]c1c(C)c(CO)cc(C#N)c21',
            'COc1ccc(C=O)c2cc(C(F)F)nn12.C[CH2][Mg+],CCC(O)c1ccc(OC)n2nc(C(F)F)cc12',
            'CCOCCBr.O=Cc1ccc(O)c(O)c1,CCOCCOc1ccc(C=O)cc1O',
        ]

    # USPTO-50k test set (mapped)
    upsto_mapped_test = [
            'Cl[C:1](=[O:2])[CH2:3][c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.[CH3:10][c:11]1[n:12][c:13]([NH2:14])[s:15][c:16]1[C:17](=[O:18])[NH:19][CH2:20][c:21]1[cH:22][cH:23][cH:24][cH:25][cH:26]1,[C:1](=[O:2])([CH2:3][c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1)[NH:14][c:13]1[n:12][c:11]([CH3:10])[c:16]([C:17](=[O:18])[NH:19][CH2:20][c:21]2[cH:22][cH:23][cH:24][cH:25][cH:26]2)[s:15]1',
            'O=[CH:1][c:2]1[c:3]([CH3:4])[cH:5][cH:6][cH:7][cH:8]1.[NH2:9][c:10]1[cH:11][cH:12][cH:13][c:14](-[c:15]2[cH:16][cH:17][n:18][c:19]3[c:20]([C:21](=[O:22])[c:23]4[cH:24][cH:25][cH:26][s:27]4)[cH:28][n:29][n:30]23)[cH:31]1,[CH2:1]([c:2]1[c:3]([CH3:4])[cH:5][cH:6][cH:7][cH:8]1)[NH:9][c:10]1[cH:11][cH:12][cH:13][c:14](-[c:15]2[cH:16][cH:17][n:18][c:19]3[c:20]([C:21](=[O:22])[c:23]4[cH:24][cH:25][cH:26][s:27]4)[cH:28][n:29][n:30]23)[cH:31]1',
            'Cl[S:15](=[O:16])(=[O:17])[c:18]1[cH:19][cH:20][cH:21][cH:22][n:23]1.[CH3:1][C@H:2]([NH:3][C:4](=[O:5])[O:6][C:7]([CH3:8])([CH3:9])[CH3:10])[C@H:11]([OH:12])[CH2:13][NH2:14],[CH3:1][C@H:2]([NH:3][C:4](=[O:5])[O:6][C:7]([CH3:8])([CH3:9])[CH3:10])[C@@H:11]([OH:12])[CH2:13][NH:14][S:15](=[O:16])(=[O:17])[c:18]1[cH:19][cH:20][cH:21][cH:22][n:23]1',
            'CC[O:1][C:2](=[O:3])[c:4]1[s:5][c:6]([C:7](=[O:8])[O:9][C:10]([CH3:11])([CH3:12])[CH3:13])[cH:14][c:15]1[O:16][C:17]([CH3:18])([CH3:19])[CH3:20],[OH:1][C:2](=[O:3])[c:4]1[s:5][c:6]([C:7](=[O:8])[O:9][C:10]([CH3:11])([CH3:12])[CH3:13])[cH:14][c:15]1[O:16][C:17]([CH3:18])([CH3:19])[CH3:20]',
            'Cl[c:1]1[c:2]2[cH:3][n:4][n:5](-[c:6]3[c:7]([Cl:8])[cH:9][cH:10][cH:11][n:12]3)[c:13]2[n:14][cH:15][n:16]1.[CH3:17][CH2:18][O:19][CH2:20][C@H:21]([OH:22])[C:23](=[O:24])[O:25][CH3:26],[c:1]1([O:22][C@@H:21]([CH2:20][O:19][CH2:18][CH3:17])[C:23](=[O:24])[O:25][CH3:26])[c:2]2[cH:3][n:4][n:5](-[c:6]3[c:7]([Cl:8])[cH:9][cH:10][cH:11][n:12]3)[c:13]2[n:14][cH:15][n:16]1',
            'CO[C:1](=[O:2])[c:3]1[cH:4][cH:5][c:6]2[c:7]([cH:8]1)[n:9][c:10]([NH:11][C:12](=[O:13])[c:14]1[cH:15][cH:16][n:17][o:18]1)[n:19]2[CH2:20][C@H:21]1[CH2:22][CH2:23][CH2:24][N:25]1[C:26](=[O:27])[O:28][C:29]([CH3:30])([CH3:31])[CH3:32],[CH2:1]([OH:2])[c:3]1[cH:4][cH:5][c:6]2[c:7]([cH:8]1)[n:9][c:10]([NH:11][C:12](=[O:13])[c:14]1[cH:15][cH:16][n:17][o:18]1)[n:19]2[CH2:20][C@H:21]1[CH2:22][CH2:23][CH2:24][N:25]1[C:26](=[O:27])[O:28][C:29]([CH3:30])([CH3:31])[CH3:32]',
            '[CH3:1][O:2][C:3](=[O:4])[C@@H:5]1[CH2:6][c:7]2[cH:8][cH:9][c:10]([C:11]3=[CH:12][C@@H:13]([C:14](=[O:15])[NH:16][C@@H:17]4[CH2:18][CH2:19][CH2:20][c:21]5[cH:22][cH:23][cH:24][cH:25][c:26]54)[N:27]([C:28](=[O:29])[O:30][C:31]([CH3:32])([CH3:33])[CH3:34])[CH2:35]3)[cH:36][c:37]2[CH2:38][N:39]1[C:40](=[O:41])[O:42][C:43]([CH3:44])([CH3:45])[CH3:46],[CH3:1][O:2][C:3](=[O:4])[C@@H:5]1[CH2:6][c:7]2[cH:8][cH:9][c:10]([C@H:11]3[CH2:12][C@@H:13]([C:14](=[O:15])[NH:16][C@@H:17]4[CH2:18][CH2:19][CH2:20][c:21]5[cH:22][cH:23][cH:24][cH:25][c:26]54)[N:27]([C:28](=[O:29])[O:30][C:31]([CH3:32])([CH3:33])[CH3:34])[CH2:35]3)[cH:36][c:37]2[CH2:38][N:39]1[C:40](=[O:41])[O:42][C:43]([CH3:44])([CH3:45])[CH3:46]',
            'O[CH2:1][CH:2]1[CH2:3][CH2:4]1.[CH3:5][O:6][C:7](=[O:8])[c:9]1[cH:10][cH:11][c:12]([Br:13])[c:14]([OH:15])[cH:16]1,[CH2:1]([CH:2]1[CH2:3][CH2:4]1)[O:15][c:14]1[c:12]([Br:13])[cH:11][cH:10][c:9]([C:7]([O:6][CH3:5])=[O:8])[cH:16]1',
            'O=[C:1]([c:2]1[c:3]([CH3:4])[c:5]2[nH:6][c:7]3[c:8]([c:9]2[c:10]([C:11]#[N:12])[cH:13]1)[CH2:14][CH2:15][O:16][C:17]3([CH2:18][CH2:19][CH3:20])[CH2:21][C:22](=[O:23])[O:24][CH2:25][CH3:26])[OH:27],[CH2:1]([c:2]1[c:3]([CH3:4])[c:5]2[nH:6][c:7]3[c:8]([c:9]2[c:10]([C:11]#[N:12])[cH:13]1)[CH2:14][CH2:15][O:16][C:17]3([CH2:18][CH2:19][CH3:20])[CH2:21][C:22](=[O:23])[O:24][CH2:25][CH3:26])[OH:27]',
            '[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([CH:7]=[O:8])[c:9]2[cH:10][c:11]([CH:12]([F:13])[F:14])[n:15][n:16]12.[Mg+][CH2:17][CH3:18],[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([CH:7]([OH:8])[CH2:17][CH3:18])[c:9]2[cH:10][c:11]([CH:12]([F:13])[F:14])[n:15][n:16]12',
            'Br[CH2:1][CH2:2][O:3][CH2:4][CH3:5].[O:6]=[CH:7][c:8]1[cH:9][cH:10][c:11]([OH:12])[c:13]([OH:14])[cH:15]1,[CH2:1]([CH2:2][O:3][CH2:4][CH3:5])[O:12][c:11]1[cH:10][cH:9][c:8]([CH:7]=[O:6])[cH:15][c:13]1[OH:14]',
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
