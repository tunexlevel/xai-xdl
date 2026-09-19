# =========Model File============
import torch
import torch.nn as nn
import math


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        emb_dim=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        pad_idx=0,
        max_len=200,
    ):
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
            batch_first=True,  # Important: Expects (Batch, Seq)
        )

        self.fc_out = nn.Linear(emb_dim, output_dim)

    def generate_square_subsequent_mask(self, sz):
        # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
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
        src_padding_mask = src == self.pad_idx
        tgt_padding_mask = tgt == self.pad_idx

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        # src: (B, src_len)
        # tgt: (B, tgt_len)

        # Create masks for the transformer
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt
        )

        # Embed and add position info
        src_emb = self.positional_encoding(self.embedding(src))
        tgt_emb = self.positional_encoding(self.embedding(tgt))

        # Transformer Pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return self.fc_out(outs)

    def generate(self, src, sos_idx, eos_idx):
        batch_size = src.shape[0]
        device = src.device

        # 1. Encode
        src_emb = self.positional_encoding(self.embedding(src))
        src_padding_mask = src == self.pad_idx
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )

        ys = torch.ones(batch_size, 1).fill_(sos_idx).type(torch.long).to(device)

        # List to store cross-attention weights for each step
        # Shape per step: (batch, n_heads, tgt_len, src_len)
        all_attention_weights = []

        for i in range(self.max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)
            tgt_padding_mask = ys == self.pad_idx
            tgt_emb = self.positional_encoding(self.embedding(ys))

            # --- MANUAL DECODER PASS TO GRAB ATTENTION ---
            output = tgt_emb
            last_layer_attn = None

            for layer in self.transformer.decoder.layers:
                output = layer.self_attn(
                    output,
                    output,
                    output,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_padding_mask,
                )[0]
                output = layer.dropout1(output)
                output = layer.norm1(output + tgt_emb)

                query = output
                output, attn_weights = layer.multihead_attn(
                    query,
                    memory,
                    memory,
                    key_padding_mask=src_padding_mask,
                    need_weights=True,
                )
                last_layer_attn = attn_weights

                output = layer.dropout2(output)
                output = layer.norm2(output + query)

                ff_output = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(output)))
                )
                output = layer.norm3(output + ff_output)

            all_attention_weights.append(last_layer_attn)
            prob = self.fc_out(output[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

            if (next_word == eos_idx).all():
                break

        combined_attn = torch.cat(
            [step[:, -1:, :] for step in all_attention_weights], dim=1
        )
        return ys, combined_attn

    def beam_search_candidates(self, src, sos_idx, eos_idx, beam_width=8, max_len=120):
        device = src.device
        src_padding_mask = src == self.pad_idx
        src_emb = self.positional_encoding(self.embedding(src))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )

        beams = [(torch.tensor([sos_idx], device=device), 0.0)]
        finished = []

        for _ in range(max_len - 1):
            candidates = []
            for seq, score in beams:
                if seq[-1].item() == eos_idx and seq.numel() > 1:
                    finished.append((seq, score))
                    continue

                tgt_mask = self.generate_square_subsequent_mask(seq.size(0)).to(device)
                tgt_padding_mask = seq == self.pad_idx
                tgt_emb = self.positional_encoding(self.embedding(seq.unsqueeze(0)))

                out = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask.unsqueeze(0),
                    memory_key_padding_mask=src_padding_mask,
                )

                logits = self.fc_out(out[:, -1])
                log_probs = torch.log_softmax(logits, dim=-1)[0]
                topk = torch.topk(log_probs, k=min(beam_width, log_probs.numel()))

                for next_idx, next_logp in zip(
                    topk.indices.tolist(), topk.values.tolist()
                ):
                    if next_idx == self.pad_idx:
                        continue
                    new_seq = torch.cat(
                        [seq, torch.tensor([next_idx], device=device)], dim=0
                    )
                    candidates.append((new_seq, score + float(next_logp)))

            if not candidates:
                break

            uniq = {}
            for seq, score in candidates:
                key = tuple(seq.cpu().tolist())
                if key not in uniq or score > uniq[key][1]:
                    uniq[key] = (seq, score)

            beams = sorted(uniq.values(), key=lambda x: x[1], reverse=True)[:beam_width]
            if len(finished) >= beam_width:
                break

        all_results = finished + [(seq, score) for seq, score in beams]
        if not all_results:
            return []
        return sorted(all_results, key=lambda x: x[1], reverse=True)

    def greedy_decode(self, src, sos_idx, eos_idx, max_len=120):
        device = src.device

        # Encode source
        src_emb = self.positional_encoding(self.embedding(src))
        src_padding_mask = src == self.pad_idx

        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )

        # Start with <sos>
        ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):

            tgt_emb = self.positional_encoding(self.embedding(ys))

            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)

            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            logits = self.fc_out(output[:, -1, :])

            # Greedy selection
            next_token = logits.argmax(dim=-1).unsqueeze(1)

            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == eos_idx:
                break

        return ys

        # ============================================================
    





# Helper class for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
