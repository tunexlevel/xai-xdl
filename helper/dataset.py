import torch
from torch.utils.data import Dataset
from helper.utils import tokenize_smiles

class ReactionDataset(Dataset):
    def __init__(self, df, token2idx, max_len=160, retrosynthesis=True):
        self.df = df
        self.token2idx = token2idx
        self.max_len = max_len
        self.retrosynthesis = retrosynthesis

        self.sos_idx = self.token2idx['<sos>']
        self.eos_idx = self.token2idx['<eos>']
        self.pad_idx = self.token2idx['<pad>']
        self.unk_idx = self.token2idx['<unk>']

    def __len__(self):
        return len(self.df)

    def encode_src(self, smiles):
        tokens = tokenize_smiles(smiles)
        # Reserve 1 slot for <eos>
        tokens = tokens[: self.max_len - 1]
        tokens = tokens + ['<eos>']
        
        ids = [self.token2idx.get(tok, self.unk_idx) for tok in tokens]
        padded = ids + [self.pad_idx] * (self.max_len - len(ids))
        return torch.tensor(padded, dtype=torch.long)

    def encode_tgt(self, smiles):
        tokens = tokenize_smiles(smiles)
        # Reserve 2 slots for <sos> and <eos>
        tokens = tokens[: self.max_len - 2]
        tokens = ['<sos>'] + tokens + ['<eos>']
        
        ids = [self.token2idx.get(tok, self.unk_idx) for tok in tokens]
        padded = ids + [self.pad_idx] * (self.max_len - len(ids))
        return torch.tensor(padded, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.retrosynthesis:
            src = self.encode_src(row['products'])
            tgt = self.encode_tgt(row['reactants'])
        else:
            src = self.encode_src(row['reactants'])
            tgt = self.encode_tgt(row['products'])
        return src, tgt