import torch
import torch.nn as nn
import json
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from model_1 import Seq2SeqTransformer
from utils import tokenize_smiles
from data_loader import load_uspto_file


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
# --- 2. HELPER: Canonicalization ---
def canonicalize_smiles(smiles):
    """
    Standardizes a SMILES string using RDKit.
    Returns an empty string if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return ""

# --- 3. CORE: BEAM SEARCH FUNCTION ---
def beam_search_predict(reactant_smiles, beam_width=10, max_len=120):
    model.eval()
    
    # Preprocess Input
    tokens = ["<sos>"] + tokenize_smiles(reactant_smiles) + ["<eos>"]
    input_ids = [token2idx.get(tok, token2idx["<unk>"]) for tok in tokens]
    input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        # --- ENCODE ---
        embedded_src = model.embedding(input_tensor)
        _, encoder_hidden = model.encoder(embedded_src)
        
        # --- BEAM SEARCH INITIALIZATION ---
        # Each candidate is a tuple: (current_sequence_of_tokens, cumulative_log_prob, current_hidden_state)
        # Start with just the <sos> token
        candidates = [([sos_idx], 0.0, encoder_hidden)]
        
        # Loop through time steps
        for _ in range(max_len):
            all_next_candidates = []
            
            # Expand each current candidate
            for seq, score, hidden in candidates:
                # If this candidate already ended with <eos>, keep it as is
                if seq[-1] == eos_idx:
                    all_next_candidates.append((seq, score, hidden))
                    continue
                
                # Prepare input for decoder (last token of the sequence)
                last_token = torch.tensor([[seq[-1]]], device=device)
                embedded_tgt = model.embedding(last_token)
                
                # Decode one step
                output, new_hidden = model.decoder(embedded_tgt, hidden)
                logits = model.fc_out(output) # (1, 1, vocab_size)
                
                # Get log probabilities (softmax + log)
                log_probs = torch.log_softmax(logits, dim=2).squeeze()
                
                # Get top 'beam_width' tokens for this specific branch
                top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                # Create new branches
                for k in range(beam_width):
                    token_id = top_k_indices[k].item()
                    token_score = top_k_log_probs[k].item()
                    
                    new_seq = seq + [token_id]
                    new_score = score + token_score # Add log prob to cumulative score
                    all_next_candidates.append((new_seq, new_score, new_hidden))
            
            # --- PRUNING ---
            # Sort all candidates by score (highest log_prob first) and keep only top k
            ordered = sorted(all_next_candidates, key=lambda x: x[1], reverse=True)
            candidates = ordered[:beam_width]
            
            # Optional: Stop early if all candidates hit <eos>
            if all(c[0][-1] == eos_idx for c in candidates):
                break
    
    # --- DECODE TO SMILES ---
    final_smiles_list = []
    for seq, score, _ in candidates:
        # Convert IDs back to tokens (exclude <sos> and <eos>)
        pred_tokens = [idx2token[idx] for idx in seq if idx not in [sos_idx, eos_idx, pad_idx]]
        smiles = "".join(pred_tokens)
        final_smiles_list.append(smiles)
        
    return final_smiles_list

# --- 4. EVALUATION LOOP FOR TOP-K ---
def evaluate_model(max_samples=100):
    df_test = load_uspto_file("data/uspto50k/raw_test.csv", max_samples)
    
    # Counters for accuracy
    top_k_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    
    print(f"Starting Evaluation on {len(df_test)} samples...")
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        reactant = row['reactants'] # Adjust column name if needed
        true_product = row['products'] # Adjust column name if needed
        
        # 1. Get True Canonical SMILES
        canon_true = canonicalize_smiles(true_product)
        if not canon_true: continue # Skip invalid ground truth
        
        # 2. Get Top-10 Predictions
        predicted_candidates = beam_search_predict(reactant, beam_width=10)
        
        # 3. Canonicalize Predictions
        canon_preds = [canonicalize_smiles(s) for s in predicted_candidates]
        
        # 4. Check Matches
        total += 1
        
        # Check Top-1
        if canon_true in canon_preds[:1]:
            top_k_hits[1] += 1
            
        # Check Top-3
        if canon_true in canon_preds[:3]:
            top_k_hits[3] += 1
            
        # Check Top-5
        if canon_true in canon_preds[:5]:
            top_k_hits[5] += 1
            
        # Check Top-10
        if canon_true in canon_preds[:10]:
            top_k_hits[10] += 1

    # --- PRINT RESULTS ---
    print("\n=== Final Top-k Accuracy ===")
    results = {}
    for k in [1, 3, 5, 10]:
        acc = (top_k_hits[k] / total) * 100
        results[f'Top-{k}'] = acc
        print(f"Top-{k}: {acc:.2f}%")
        
    return results

# --- RUN IT ---
# Uncomment below to run
evaluate_model(max_samples=100)