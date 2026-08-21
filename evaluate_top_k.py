import math
import torch
import json
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from model_1 import Seq2SeqTransformer
from utils import build_vocab, tokenize_smiles
from data_loader import load_uspto_file
from rdkit import RDLogger
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SETUP (Load Model and Vocab as before) ---
# Assuming 'model', 'token2idx', 'idx2token', 'device' are loaded/defined as in your original code
# Make sure to import your Seq2SeqGRU class here or ensure it's in scope


# === Load vocab and model ===
with open("token2idx.json") as f:
    token2idx = json.load(f)
with open("idx2token.json") as f:
    idx2token = {int(k): v for k, v in json.load(f).items()}


# Hyperparameters for Evaluation
BEAM_WIDTH = 10  # This effectively gives you Top-10 candidates
MAX_LEN = 120
sos_idx = token2idx["<sos>"]
eos_idx = token2idx["<eos>"]
pad_idx = token2idx["<pad>"]

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
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles: str) -> str:
    """
    Safely canonicalizes a SMILES string.
    Returns "" if:
        - input is None
        - input is empty or whitespace
        - input is NaN
        - RDKit cannot parse it
    """
    # Reject None, NaN, empty, whitespace
    if smiles is None:
        return ""

    if isinstance(smiles, float) and math.isnan(smiles):
        return ""

    smiles = smiles.strip()
    if smiles == "":
        return ""

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return ""

# --- 3. CORE: BEAM SEARCH FUNCTION ---
def beam_search_predict(reactant_smiles, beam_width=10, max_len=120):
    model.eval()
    device = next(model.parameters()).device # Get model's device automatically
    
    # 1. PREPARE INPUT
    tokens = ["<sos>"] + tokenize_smiles(reactant_smiles) + ["<eos>"]
    unk_idx = token2idx.get("<unk>", 0)
    src_ids = [token2idx.get(tok, unk_idx) for tok in tokens]
    
    # Shape: (1, src_len)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    # 2. CREATE MASKS FOR ENCODER
    # Your model's create_mask method expects 2 args, but we only have src here.
    # We'll build the src masks manually to be safe.
    src_padding_mask = (src_tensor == model.pad_idx)
    src_mask = torch.zeros((src_tensor.shape[1], src_tensor.shape[1]), device=device).type(torch.bool)

    with torch.no_grad():
        # --- ENCODER STEP (Run Once) ---
        # We encode the reactant ONCE. This creates the "memory".
        src_emb = model.positional_encoding(model.embedding(src_tensor))
        
        # Call the encoder inside nn.Transformer
        memory = model.transformer.encoder(
            src_emb, 
            mask=src_mask, 
            src_key_padding_mask=src_padding_mask
        ) # Shape: (1, src_len, emb_dim)

        # EXPAND MEMORY: We need to replicate this memory for every beam candidate
        # New Shape: (beam_width, src_len, emb_dim)
        memory = memory.expand(beam_width, -1, -1)
        
        # Also expand the padding mask because the decoder needs it for cross-attention
        src_padding_mask = src_padding_mask.expand(beam_width, -1)

        # --- BEAM SEARCH INITIALIZATION ---
        # Candidates: List of tuples (list_of_token_ids, cumulative_log_prob)
        # Start with ONE candidate: [<sos>] with score 0.0
        sos_idx = token2idx["<sos>"]
        candidates = [([sos_idx], 0.0)]
        
        # LOOP THROUGH TIME STEPS
        for i in range(max_len):
            all_next_candidates = []
            
            # 1. Filter completed vs active candidates
            active_candidates = []
            finished_candidates = []
            
            for seq, score in candidates:
                if seq[-1] == token2idx["<eos>"]:
                    finished_candidates.append((seq, score))
                else:
                    active_candidates.append((seq, score))
            
            # If all beams are finished, stop
            if not active_candidates:
                candidates = finished_candidates
                break
            
            # 2. Prepare Batch for Active Candidates
            # We feed ALL active beams into the decoder at once (Batching)
            curr_batch_size = len(active_candidates)
            
            # Slice memory to match current batch size (e.g., if we only have 3 active beams)
            curr_memory = memory[:curr_batch_size]
            curr_src_key_padding_mask = src_padding_mask[:curr_batch_size]
            
            # Create tensor of current sequences
            # Shape: (curr_batch_size, current_seq_len)
            tgt_seqs = [c[0] for c in active_candidates]
            tgt_tensor = torch.LongTensor(tgt_seqs).to(device)
            
            # 3. RUN DECODER
            # Create Causal Mask (prevent seeing future)
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            tgt_padding_mask = (tgt_tensor == model.pad_idx)
            
            # Embed
            tgt_emb = model.positional_encoding(model.embedding(tgt_tensor))
            
            # Pass through Decoder
            # Note: We access .decoder directly from nn.Transformer
            out = model.transformer.decoder(
                tgt_emb, 
                curr_memory, 
                tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=curr_src_key_padding_mask
            )
            
            # 4. GET PREDICTIONS
            # We only care about the LAST token's output
            last_step_out = out[:, -1, :] # (batch, emb_dim)
            logits = model.fc_out(last_step_out) # (batch, vocab_size)
            
            log_probs = torch.log_softmax(logits, dim=1)
            
            # 5. EXPAND BEAMS
            # Get Top-K for this step
            # We take top-k from the *entire batch* of possibilities for this step
            top_k_probs, top_k_ids = torch.topk(log_probs, beam_width)
            
            for batch_idx in range(curr_batch_size):
                parent_seq, parent_score = active_candidates[batch_idx]
                
                for k in range(beam_width):
                    token_id = top_k_ids[batch_idx, k].item()
                    token_score = top_k_probs[batch_idx, k].item()
                    
                    new_seq = parent_seq + [token_id]
                    new_score = parent_score + token_score
                    
                    all_next_candidates.append((new_seq, new_score))
            
            # 6. PRUNE
            # Combine new active candidates with finished ones
            all_candidates = all_next_candidates + finished_candidates
            # Sort by score (highest is better for log_probs, as they are negative)
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            # Keep top K
            candidates = ordered[:beam_width]
            
    # --- DECODE TO SMILES ---
    final_smiles_list = []
    for seq, score in candidates:
        # Convert IDs to Tokens (remove special tokens)
        pred_tokens = [idx2token[idx] for idx in seq if idx not in [token2idx["<sos>"], token2idx["<eos>"], token2idx["<pad>"]]]
        smiles = "".join(pred_tokens)
        final_smiles_list.append(smiles)
        
    return final_smiles_list

# --- 4. EVALUATION LOOP FOR TOP-K ---

def evaluate_model(test_csv_path, max_samples=1000):
    # df_test = pd.read_csv(test_csv_path).head(max_samples) 
    # Use your specific loader:
    df_test = load_uspto_file("data/uspto50k/raw_test.csv", max_samples=100)
    
    # Counters for accuracy
    top_k_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    
    # Variable to track total time spent only on prediction
    total_inference_time = 0.0 
    
    print(f"Starting Evaluation on {len(df_test)} samples...")
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        reactant = row['reactants'] 
        true_product = row['products'] 
        
        # 1. Get True Canonical SMILES
        canon_true = canonicalize_smiles(true_product)
        if not canon_true: 
            print("Invalid ground truth SMILES, skipping...")
            continue 
        
        # --- START TIMING ---
        # We use perf_counter() for high precision timing
        start_time = time.perf_counter()
        
        # 2. Get Top-10 Predictions (This is the "AI Service" part)
        predicted_candidates = beam_search_predict(reactant, beam_width=10)
        
        # --- END TIMING ---
        end_time = time.perf_counter()
        
        # Add the duration of this specific prediction to total
        total_inference_time += (end_time - start_time)
        
        # 3. Canonicalize Predictions
        canon_preds = [canonicalize_smiles(s) for s in predicted_candidates]
        
        # 4. Check Matches
        total += 1
        
        if canon_true in canon_preds[:1]: top_k_hits[1] += 1
        if canon_true in canon_preds[:3]: top_k_hits[3] += 1
        if canon_true in canon_preds[:5]: top_k_hits[5] += 1
        if canon_true in canon_preds[:10]: top_k_hits[10] += 1

    if total == 0:
        print("No valid samples to evaluate.")
        return {}

    # --- CALCULATE METRICS ---
    print("\n=== Final Results ===")
    results = {}
    
    # Accuracy Metrics
    for k in [1, 3, 5, 10]:
        acc = (top_k_hits[k] / total) * 100
        results[f'Top-{k}'] = acc
        print(f"Top-{k}: {acc:.2f}%")
    
    # Operational Metrics (Latency)
    avg_latency_sec = total_inference_time / total
    avg_latency_ms = avg_latency_sec * 1000  # Convert seconds to milliseconds
    
    print("-" * 30)
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Latency per Molecule: {avg_latency_ms:.2f} ms")
    print("-" * 30)
    
    results['avg_latency_ms'] = avg_latency_ms
        
    return results
# --- RUN IT ---
# Uncomment below to run
results = evaluate_model("data/uspto50k/raw_test.csv", max_samples=6000)