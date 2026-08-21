import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import re

# -------------------
# Tokenizer & Vocab
# -------------------
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def tokenize_smiles(smiles):
    # Simple tokenizer: each uppercase letter or symbol is a token
    return re.findall(r"Br|Cl|[A-Z][a-z]?|[0-9]|=|#|\(|\)|\.|\+|\-|/", smiles)

def build_vocab(smiles_list):
    tokens = []
    for smi in smiles_list:
        tokens.extend(tokenize_smiles(smi))
    tokens = list(set(tokens))
    
    token2idx = {PAD_TOKEN:0, SOS_TOKEN:1, EOS_TOKEN:2, UNK_TOKEN:3}
    idx2token = {0:PAD_TOKEN, 1:SOS_TOKEN, 2:EOS_TOKEN, 3:UNK_TOKEN}
    for t in tokens:
        idx = len(token2idx)
        token2idx[t] = idx
        idx2token[idx] = t
    return token2idx, idx2token

# -------------------
# Dataset
# -------------------
class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, token2idx, max_len=10):
        self.smiles_list = smiles_list
        self.token2idx = token2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        tokens = [SOS_TOKEN] + tokenize_smiles(smi) + [EOS_TOKEN]
        ids = [self.token2idx.get(t, self.token2idx[UNK_TOKEN]) for t in tokens]
        # pad
        if len(ids) < self.max_len:
            ids += [self.token2idx[PAD_TOKEN]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        input_ids = torch.LongTensor(ids)
        target_ids = torch.LongTensor(ids)
        return input_ids, target_ids

# -------------------
# Model
# -------------------
class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=16, hidden_dim=32, pad_idx=0, sos_idx=1, eos_idx=2, max_len=10):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.encoder = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        # Teacher forcing
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)
        embedded_tgt = self.embedding(tgt)
        outputs, _ = self.decoder(embedded_tgt, hidden)
        logits = self.fc_out(outputs)
        return logits

    def generate(self, src):
        batch_size = src.size(0)
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)
        decoder_input = torch.full((batch_size,1), self.sos_idx, dtype=torch.long, device=src.device)
        outputs = []

        for _ in range(self.max_len):
            embedded = self.embedding(decoder_input)
            out, hidden = self.decoder(embedded, hidden)
            logits = self.fc_out(out)
            next_token = logits.argmax(-1)
            
            if(next_token == self.eos_idx).all():
                break
            
            outputs.append(next_token)
            decoder_input = next_token
        outputs = torch.cat(outputs, dim=1)
        return outputs

# -------------------
# Training
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiny dataset with single molecule
all_smiles = ["CCOCCO"]
token2idx, idx2token = build_vocab(all_smiles)

dataset = MoleculeDataset(all_smiles, token2idx, max_len=5)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = Seq2SeqGRU(len(token2idx), len(token2idx), emb_dim=16, hidden_dim=32,
                   pad_idx=token2idx[PAD_TOKEN], max_len=5).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=token2idx[PAD_TOKEN])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):  # train long enough
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        output = output.reshape(-1, output.shape[-1])
        target = tgt[:,1:].reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1} loss: {total_loss:.4f}")

# -------------------
# Test
# -------------------
model.eval()
for smi in all_smiles:
    tokens = [SOS_TOKEN] + tokenize_smiles(smi) + [EOS_TOKEN]
    ids = [token2idx.get(t, token2idx[UNK_TOKEN]) for t in tokens]
    src = torch.LongTensor(ids).unsqueeze(0).to(device)
    out_ids = model.generate(src)[0].cpu().numpy()
    pred_tokens = [idx2token[i] for i in out_ids if i not in [token2idx[PAD_TOKEN], token2idx[SOS_TOKEN], token2idx[EOS_TOKEN]]]
    pred_smi = "".join(pred_tokens)
    print(f"Input: {smi}, Predicted: {pred_smi}")
