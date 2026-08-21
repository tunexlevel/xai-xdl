import torch
import torch.nn as nn

class Seq2SeqGRU2(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=128, hidden_dim=256, num_layers=1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.encoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)

        # Embed input and target sequences
        embedded_src = self.embedding(src)  # (batch_size, src_len, emb_dim)
        embedded_tgt = self.embedding(tgt)

        # Encoder
        _, hidden = self.encoder(embedded_src)

        # Decoder (teacher forcing)
        output, _ = self.decoder(embedded_tgt, hidden)

        # Output layer
        predictions = self.fc_out(output)  # (batch_size, tgt_len, output_dim)
        return predictions
