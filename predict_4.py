import torch
import json
import re
from rdkit import Chem
from utils import tokenize_smiles
from model_2 import Seq2SeqTransformer
from rdkit.Chem import Draw


# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Vocabulary
try:
    with open("token2idx.json", "r") as f:
        token2idx = json.load(f)
    with open("idx2token.json", "r") as f:
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
    model.load_state_dict(torch.load("reaction_model.pt", map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: reaction_model.pt not found.")
    exit()


# 5. Prediction Logic
def predict_product(reactant_smiles):
    model.eval()
    
    # STEP A: CLEAN THE INPUT
    # We remove the :1, :2 tags so the model sees "N" instead of "[N+:1]"
    print(f"Original: {reactant_smiles}")
    
    
    # STEP B: TOKENIZE
    tokens = tokenize_smiles(reactant_smiles)
    
    # STEP C: CONVERT TO IDS
    # If a token is still unknown, we print a warning
    src_ids = []
    for t in tokens:
        if t in token2idx:
            src_ids.append(token2idx[t])
        else:
            print(f"⚠️ Warning: Token '{t}' is unknown to the model!")
            src_ids.append(token2idx["<unk>"])
            
    print (f"Token IDs: {src_ids}")
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # STEP D: GENERATE
    with torch.no_grad():
        output, attn_weights = model.generate_with_attention(src_tensor, sos_idx, eos_idx)
        
    # DECODE OUTPUT
    output_indices = output.squeeze().tolist()
    result_tokens = []
    for idx in output_indices:
        if idx == sos_idx: continue
        if idx == eos_idx: break
        if idx == pad_idx: continue
        result_tokens.append(idx2token.get(idx, ""))

    # PREPARE ATTENTION DATA
    # attn_weights shape is (1, target_len, source_len)
    # We slice it to match the actual generated tokens length
    actual_attn = attn_weights.squeeze(0)[:len(result_tokens), :len(tokens)]
    
    return {
        "product": "".join(result_tokens),
        "data":{
            "target_tokens": result_tokens,
            "source_tokens": tokens,
            "matrix": actual_attn.cpu().tolist() # Convert to nested list for JSON
        }
    }

# === RUN IT ===
if __name__ == "__main__":
    # Your problematic input
    input_reactant = "O=[N+:1]([O-])[c:2]1[cH:3][c:4]([Cl:5])[cH:6][n:7][c:8]1[F:9]"
    
    prediction = predict_product(input_reactant)
    
    print(prediction)
    
    # print("\nPredicted Product:", prediction)
    
    # try:
    #     mol1 = Chem.MolFromSmiles(input_reactant)
    #     mol2 = Chem.MolFromSmiles(prediction)
    #     Draw.MolsToGridImage([mol1, mol2], legends=["Reactant", "Predicted Product"]).show()
    # except Exception as e:
    #     print("Visualization failed:", e)