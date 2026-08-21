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
            
    # --- DECODE TO SMILES AND NORMALIZE SCORE ---
    final_results = []
    alpha = 0.6 # Your length normalization penalty
    
    for seq, raw_score in candidates:
        # 1. Determine Sequence Length (L) for normalization
        # Exclude <sos> and <eos> for effective sequence length
        L = len([idx for idx in seq if idx not in [token2idx["<sos>"], token2idx["<eos>"], token2idx["<pad>"]]])
        
        # 2. Apply Length Normalization (Score / L^alpha)
        # Handle L=0 case if necessary, but for chemical strings L > 0
        if L > 0:
            normalized_score = raw_score / (L ** alpha)
        else:
            normalized_score = raw_score

        # Convert IDs to Tokens
        pred_tokens = [idx2token[idx] for idx in seq if idx not in [token2idx["<sos>"], token2idx["<eos>"], token2idx["<pad>"]]]
        smiles = "".join(pred_tokens)
        
        final_results.append({
            'smiles': smiles,
            'normalized_score': normalized_score,
            'raw_score': raw_score,
            'length': L
        })
        
    # Return the full structure so you can look up the score for the ground truth SMILES
    return final_results

# --- 4. EVALUATION LOOP FOR TOP-K ---

def analyze_single_reaction(reactant_smiles, true_product_smiles, beam_width=10):
    """
    Runs beam search for a single reaction and returns the rank and normalized
    confidence score of the true product within the top-K predictions.
    
    Args:
        reactant_smiles (str): The reactant SMILES string (input).
        true_product_smiles (str): The true product SMILES string (ground truth).
        beam_width (int): The beam width to use for prediction (e.g., 10).
        
    Returns:
        dict: Contains the rank and normalized score of the true product, 
              or a status if not found.
    """
    
    # 1. Canonicalize the Ground Truth
    canon_true = canonicalize_smiles(true_product_smiles)
    if not canon_true:
        return {"status": "Error: Invalid ground truth SMILES provided."}
        
    # 2. Get Top-K Predictions (This returns the list of dicts)
    predicted_candidates = beam_search_predict(reactant_smiles, beam_width=beam_width)
    
    print(f"Top-{beam_width} Predictions for Reactant: {reactant_smiles}")
    print(f"{'Rank':<5} {'SMILES':<50} {'Norm Score':<12} {'Raw Score':<10} {'Length':<6}")
    for idx, candidate in enumerate(predicted_candidates):
        print(f"{idx+1:<5} {candidate['smiles']:<50} {candidate['normalized_score']:<12.4f} {candidate['raw_score']:<10.4f} {candidate['length']:<6}")  
    
    # 3. Prepare Lists for Comparison
    # We extract the SMILES and the score lists, maintaining their rank order
    smiles_only_list = [d['smiles'] for d in predicted_candidates]
    norm_scores_list = [d['normalized_score'] for d in predicted_candidates]
    
    # 4. Canonicalize Predictions (to match the ground truth format)
    canon_preds = [canonicalize_smiles(s) for s in smiles_only_list]
    
    # 5. Locate the True Product and Extract Score
    try:
        # Find the index (rank-1) of the true product
        rank_index = canon_preds.index(canon_true)
        
        # Score and rank are at the same index
        target_normalized_score = norm_scores_list[rank_index]
        
        return {
            "reactant": reactant_smiles,
            "product": true_product_smiles,
            "predicted_rank": rank_index + 1, # Convert 0-based index to 1-based rank
            "normalized_score": target_normalized_score,
            "status": "Success"
        }
        
    except ValueError:
        # True product was not found in the top-K beam
        return {
            "reactant": reactant_smiles,
            "product": true_product_smiles,
            "status": f"Failure: True product not found in Top {beam_width}."
        }


def beam_search_predict_with_weight(reactant_smiles, beam_width=10, max_len=120):
    model.eval()
    device = next(model.parameters()).device
    
    # 1. PREPARE INPUT
    src_tokens = tokenize_smiles(reactant_smiles)
    tokens = ["<sos>"] + src_tokens + ["<eos>"]
    src_ids = [token2idx.get(tok, token2idx.get("<unk>", 0)) for tok in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    src_padding_mask = (src_tensor == model.pad_idx)
    src_mask = torch.zeros((src_tensor.shape[1], src_tensor.shape[1]), device=device).bool()

    with torch.no_grad():
        # --- ENCODER STEP ---
        src_emb = model.positional_encoding(model.embedding(src_tensor))
        memory = model.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # --- BEAM SEARCH INITIALIZATION ---
        # Candidates: (seq, score, attention_history)
        sos_idx = token2idx["<sos>"]
        candidates = [([sos_idx], 0.0, [])] 
        
        for i in range(max_len):
            all_next_candidates = []
            active_candidates = [c for c in candidates if c[0][-1] != token2idx["<eos>"]]
            finished_candidates = [c for c in candidates if c[0][-1] == token2idx["<eos>"]]
            
            if not active_candidates:
                break
            
            # Prepare batch for decoder
            curr_batch_size = len(active_candidates)
            curr_memory = memory.expand(curr_batch_size, -1, -1)
            curr_src_padding_mask = src_padding_mask.expand(curr_batch_size, -1)
            
            tgt_seqs = [c[0] for c in active_candidates]
            tgt_tensor = torch.LongTensor(tgt_seqs).to(device)
            
            # --- MANUAL DECODER PASS TO GET ATTENTION ---
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            tgt_padding_mask = (tgt_tensor == model.pad_idx)
            tgt_emb = model.positional_encoding(model.embedding(tgt_tensor))
            
            # Instead of model.transformer.decoder(...), we iterate layers
            # to extract cross-attention weights
            decoder_output = tgt_emb
            last_layer_attn = None
            
            for layer in model.transformer.decoder.layers:
                # 1. Self-attention
                decoder_output = layer.self_attn(decoder_output, decoder_output, decoder_output, 
                                                attn_mask=tgt_mask, 
                                                key_padding_mask=tgt_padding_mask)[0]
                decoder_output = layer.norm1(decoder_output + layer.dropout1(decoder_output))
                
                # 2. Cross-attention (THIS EXTRACTS THE WEIGHTS)
                query = decoder_output
                decoder_output, attn_weights = layer.multihead_attn(
                    query, curr_memory, curr_memory,
                    key_padding_mask=curr_src_padding_mask,
                    need_weights=True
                )
                last_layer_attn = attn_weights # Shape: (batch, tgt_len, src_len)
                decoder_output = layer.norm2(decoder_output + layer.dropout2(decoder_output))
                
                # 3. Feed Forward
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(decoder_output))))
                decoder_output = layer.norm3(decoder_output + layer.dropout(ff_output))

            # Get logits for the LAST step
            logits = model.fc_out(decoder_output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=1)
            
            # Extract the attention for the NEWLY generated token (last query step)
            # Shape: (batch, 1, src_len)
            step_attn = last_layer_attn[:, -1:, :] 
            
            # Expand Beams
            top_k_probs, top_k_ids = torch.topk(log_probs, beam_width)
            
            for batch_idx in range(curr_batch_size):
                parent_seq, parent_score, parent_attn_hist = active_candidates[batch_idx]
                
                # Each child inherits the attention history of its parent plus this step
                current_token_attn = step_attn[batch_idx] # (1, src_len)
                
                for k in range(beam_width):
                    token_id = top_k_ids[batch_idx, k].item()
                    new_attn_hist = parent_attn_hist + [current_token_attn]
                    
                    all_next_candidates.append((
                        parent_seq + [token_id],
                        parent_score + top_k_probs[batch_idx, k].item(),
                        new_attn_hist
                    ))
            
            # Sort and Prune
            all_candidates = all_next_candidates + finished_candidates
            candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    # --- FINAL PROCESSING ---
    print(f"Top-{beam_width} Candidates:")
    for idx, (seq, score, attn_hist) in enumerate(candidates):
        pred_tokens = [idx2token[idx] for idx in seq if idx not in [token2idx["<sos>"], token2idx["<eos>"], token2idx["<pad>"]]]
        print(f"Rank {idx+1}: SMILES: {''.join(pred_tokens)}, Score: {score:.4f}, Attn Steps: {len(attn_hist)}")
    
    # Concatenate the list of 1-row attention weights into a matrix
    # Shape: (final_tgt_len, src_len)
    final_attn_matrix = torch.cat(best_attn_list, dim=0).cpu().numpy()

    # Normalize tokens for result
    pred_tokens = [idx2token[idx] for idx in best_seq if idx not in [token2idx["<sos>"], token2idx["<eos>"], token2idx["<pad>"]]]
    
    return {
        "smiles": "".join(pred_tokens),
        "target_tokens": pred_tokens,
        # "source_tokens": src_tokens,
        # "attention_matrix": final_attn_matrix.tolist()
    }
    

def beam_search_predict3(reactant_smiles, beam_width=10, max_len=120):
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Prepare Input
    src_tokens = tokenize_smiles(reactant_smiles)
    # Ensure tokens are clean
    src_ids = [token2idx.get("<sos>")] + [token2idx.get(tok, token2idx.get("<unk>")) for tok in src_tokens] + [token2idx.get("<eos>")]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    src_padding_mask = (src_tensor == model.pad_idx) # (1, src_len)

    with torch.no_grad():
        src_emb = model.positional_encoding(model.embedding(src_tensor))
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        sos_idx = token2idx["<sos>"]
        eos_idx = token2idx["<eos>"]
        
        # (sequence, score, attention_list)
        candidates = [([sos_idx], 0.0, [])]
        
        for i in range(max_len):
            all_next_candidates = []
            active_candidates = [c for c in candidates if c[0][-1] != eos_idx]
            finished_candidates = [c for c in candidates if c[0][-1] == eos_idx]
            
            if not active_candidates: break
            
            # Batching active beams
            curr_batch_size = len(active_candidates)
            curr_memory = memory.expand(curr_batch_size, -1, -1)
            curr_src_mask = src_padding_mask.expand(curr_batch_size, -1)
            
            tgt_seqs = [c[0] for c in active_candidates]
            tgt_tensor = torch.LongTensor(tgt_seqs).to(device)
            
            # --- Air-Tight Masking ---
            # IMPORTANT: mask must be (T, T) where T is current seq len
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            
            tgt_emb = model.positional_encoding(model.embedding(tgt_tensor))
            
            # Manual Decoder Pass
            decoder_output = tgt_emb
            last_attn = None
            
            for layer in model.transformer.decoder.layers:
                # Self Attn
                decoder_output = layer.self_attn(decoder_output, decoder_output, decoder_output, 
                                                attn_mask=tgt_mask)[0] # Removed key_padding_mask for stability
                decoder_output = layer.norm1(decoder_output + layer.dropout1(decoder_output))
                
                # Cross Attn
                query = decoder_output
                decoder_output, attn_weights = layer.multihead_attn(
                    query, curr_memory, curr_memory,
                    key_padding_mask=curr_src_mask,
                    need_weights=True
                )
                last_attn = attn_weights
                decoder_output = layer.norm2(decoder_output + layer.dropout2(decoder_output))
                
                # FF
                ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(decoder_output))))
                decoder_output = layer.norm3(decoder_output + layer.dropout(ff_out))

            logits = model.fc_out(decoder_output[:, -1, :])
            
            # --- Repetition Penalty ---
            # Penalize tokens already in the sequence to prevent [O:9][O:9]
            for b_idx in range(curr_batch_size):
                for seen_token in set(tgt_seqs[b_idx]):
                    if seen_token not in [sos_idx, token2idx.get("(", 0), token2idx.get(")", 0)]:
                        logits[b_idx, seen_token] -= 2.0 # Adjust penalty strength as needed

            log_probs = torch.log_softmax(logits, dim=1)
            step_attn = last_attn[:, -1:, :] 
            
            top_k_probs, top_k_ids = torch.topk(log_probs, beam_width)
            
            for b_idx in range(curr_batch_size):
                p_seq, p_score, p_attn = active_candidates[b_idx]
                for k in range(beam_width):
                    t_id = top_k_ids[b_idx, k].item()
                    new_next_attn = p_attn + [step_attn[b_idx]]
                    all_next_candidates.append((p_seq + [t_id], p_score + top_k_probs[b_idx, k].item(), new_next_attn))
            
            candidates = sorted(all_next_candidates + finished_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Final return as before...
        # --- FINAL PROCESSING ---
        print(f"Top-{beam_width} Candidates:")
        for idx, (seq, score, attn_hist) in enumerate(candidates):
            pred_tokens = [idx2token[idx] for idx in seq if idx not in [token2idx["<sos>"], token2idx["<eos>"], token2idx["<pad>"]]]
            print(f"Rank {idx+1}: SMILES: {''.join(pred_tokens)}, Score: {score:.4f}, Attn Steps: {len(attn_hist)}")
        
        # # Concatenate the list of 1-row attention weights into a matrix
        # # Shape: (final_tgt_len, src_len)
        # final_attn_matrix = torch.cat(best_attn_list, dim=0).cpu().numpy()

        # # Normalize tokens for result
        # pred_tokens = [idx2token[idx] for idx in best_seq if idx not in [token2idx["<sos>"], token2idx["<eos>"], token2idx["<pad>"]]]
        
        # return {
        #     "smiles": "".join(pred_tokens),
        #     "target_tokens": pred_tokens,
        #     # "source_tokens": src_tokens,
        #     # "attention_matrix": final_attn_matrix.tolist()
        # }

# --- RUN IT ---
# Uncomment below to run
suzuki_reactant = "O=C1CCC(=O)N1[Br:1].[CH3:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]" # (Example SMILES - use your exact string)
suzuki_product = "[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]" # (Example SMILES - use your exact string)

result = beam_search_predict3(suzuki_reactant, beam_width=10)

print(result)


#result = analyze_single_reaction(suzuki_reactant, suzuki_product, beam_width=10)

#print(f"Result: Rank {result['predicted_rank']} | Score {result['normalized_score']:.4f}")