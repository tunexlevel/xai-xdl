import torch
import json
from utils import tokenize_smiles
from data_loader import load_uspto_file
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from model_1 import Seq2SeqTransformer

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

model.load_state_dict(torch.load("reaction_model.pt", map_location=device))
model.to(device)
model.eval()

# === Prediction function ===
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
            
    #print (f"Token IDs: {src_ids}")
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # STEP D: GENERATE
    with torch.no_grad():
        output = model.generate(src_tensor, sos_idx, eos_idx)
        
    # STEP E: DECODE
    output_indices = output.squeeze().tolist()
    if isinstance(output_indices, int): output_indices = [output_indices] # Handle single token case
    
    #print(f"Output Token IDs: {output_indices}")
    
    result_tokens = []
    for idx in output_indices:
        if idx == sos_idx: continue
        if idx == eos_idx: break
        if idx == pad_idx: continue
        result_tokens.append(idx2token.get(idx, ""))
        
    return "".join(result_tokens)

# === Example usage ===
if __name__ == "__main__":
    df = load_uspto_file("data/uspto50k/raw_test.csv", max_samples=100)
    total_samples = len(df)
    print(f"Total samples to predict: {total_samples}\n")
    accuracy_count = 0
    for i, row in df.iterrows():
        reactant = row["reactants"]
        true_product = row["products"]
        predicted_product = predict_product(reactant)

        print(f"Reactant:  {reactant}")
        print(f"True Product: {true_product}")
        print(f"Predicted Product: {predicted_product}")
        
        if true_product == predicted_product:
            accuracy_count += 1
            print("Prediction Correct!")
        else:
            print("Prediction Incorrect.")
        print("-" * 50) 
        
    accuracy = accuracy_count / total_samples * 100
    print(f"Prediction Accuracy: {accuracy:.2f}%")
    

   
