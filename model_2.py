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

    def generate_with_attention(self, src, sos_idx, eos_idx):
        batch_size = src.shape[0]
        device = src.device
        
        # 1. Setup a container to catch the weights
        captured_weights = []

        def hook_fn(module, input, output):
            # The second element of multihead_attn output is the weights
            # (if need_weights=True was set during the call)
            # Note: Standard nn.TransformerDecoder usually doesn't return weights, 
            # so we hook the specific multihead_attn layer inside the last decoder layer.
            pass

        # Better approach: Temporarily patch the last layer to return weights
        # Or, manually step through layers but use the LAYER objects directly:
        
        src_padding_mask = (src == self.pad_idx)
        src_emb = self.positional_encoding(self.embedding(src))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        ys = torch.ones(batch_size, 1).fill_(sos_idx).type(torch.long).to(device)
        all_attn = []

        for i in range(self.max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)
            tgt_padding_mask = (ys == self.pad_idx)
            tgt_emb = self.positional_encoding(self.embedding(ys))
            
            # --- THE TRICK ---
            # Instead of self.transformer.decoder(tgt_emb, memory...), 
            # we loop through the layers that ALREADY EXIST in your model.
            
            output = tgt_emb
            for layer_idx, layer in enumerate(self.transformer.decoder.layers):
                # Standard forward pass using the layer's internal logic
                # We pass 'need_weights=True' to the cross-attention call
                
                # 1. Self-attention
                output2 = layer.self_attn(output, output, output, attn_mask=tgt_mask,
                                        key_padding_mask=tgt_padding_mask)[0]
                output = layer.norm1(output + layer.dropout1(output2))
                
                # 2. Cross-attention (Grab weights here!)
                # We call the internal multihead_attn directly
                output2, attn_weights = layer.multihead_attn(output, memory, memory,
                                                            key_padding_mask=src_padding_mask,
                                                            need_weights=True)
                output = layer.norm2(output + layer.dropout2(output2))
                
                # 3. Feed Forward
                output2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
                output = layer.norm3(output + layer.dropout(output2))
                
                # Save weights from the last layer
                if layer_idx == len(self.transformer.decoder.layers) - 1:
                    all_attn.append(attn_weights[:, -1:, :]) # (batch, 1, src_len)

            prob = self.fc_out(output[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            
            if (next_word == eos_idx).all():
                break

        return ys, torch.cat(all_attn, dim=1)
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
    
    
    