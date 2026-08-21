#consider my code, I want to get the attention weights from the transformer model during generation for debugging purposes. I have added some placeholder code in predict.py to compute and print the average attention weights, but I am not sure how to modify the model's generate function to return these weights. Can you help me with that?

#model.py page

import torch
import torch.nn as nn
import math

class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=256, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, pad_idx=0, max_len=200):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx
        self.max_len = max_len

        # Embeddings + Positional Encoding
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(emb_dim, dropout, max_len)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: Expects (Batch, Seq)
        )

        self.fc_out = nn.Linear(emb_dim, output_dim)

    def generate_square_subsequent_mask(self, sz):
        #mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # Generate the causal boolean mask (True = ignore future)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Source mask is usually all False (allow everything) unless you want to mask something specific
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).bool()

        # Padding masks (True = ignore padding)
        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        # src: (B, src_len)
        # tgt: (B, tgt_len)
        
        # Create masks for the transformer
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        
        # Embed and add position info
        src_emb = self.positional_encoding(self.embedding(src))
        tgt_emb = self.positional_encoding(self.embedding(tgt))
        
        # Transformer Pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return self.fc_out(outs)

    def generate(self, src, sos_idx, eos_idx):
        """
        Greedy decoding for inference
        """
        batch_size = src.shape[0]
        device = src.device
        
        # Encode
        src_mask = torch.zeros((src.shape[1], src.shape[1]), device=device).type(torch.bool)
        src_padding_mask = (src == self.pad_idx)
        
        src_emb = self.positional_encoding(self.embedding(src))
        memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        # Start with SOS
        ys = torch.ones(batch_size, 1).fill_(sos_idx).type(torch.long).to(device)
        
        for i in range(self.max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)
            tgt_padding_mask = (ys == self.pad_idx)
            
            tgt_emb = self.positional_encoding(self.embedding(ys))
            
            # Decode using memory from encoder
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                                           tgt_key_padding_mask=tgt_padding_mask,
                                           memory_key_padding_mask=src_padding_mask)
            
            prob = self.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            
            # Simple break if all items in batch have EOS (optional optimization)
            # This is a simplified check; usually done per-item
            if (next_word == eos_idx).all():
                break
                
        return ys

# Helper class for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    


#predict.py page
import torch
import json
import re
from rdkit import Chem
from utils import tokenize_smiles
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
        output = model.generate(src_tensor, sos_idx, eos_idx)
        
    # STEP E: DECODE
    output_indices = output.squeeze().tolist()
    if isinstance(output_indices, int): output_indices = [output_indices] # Handle single token case
    
    print(f"Output Token IDs: {output_indices}")
    
    #Attention Debug
    # avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # (src_len, tgt_len)
    # print(f"Avg Attention Shape: {avg_attention.shape}")
    # print(f"Avg Attention (first 5 rows):\n{avg_attention[:5,:5]}")
    
    result_tokens = []
    for idx in output_indices:
        if idx == sos_idx: continue
        if idx == eos_idx: break
        if idx == pad_idx: continue
        result_tokens.append(idx2token.get(idx, ""))
        
    return "".join(result_tokens)

# === RUN IT ===
if __name__ == "__main__":
    # Your problematic input
    input_reactant = "O=[N+:1]([O-])[c:2]1[cH:3][c:4]([Cl:5])[cH:6][n:7][c:8]1[F:9]"
    
    prediction = predict_product(input_reactant)
    print("\nPredicted Product:", prediction)
