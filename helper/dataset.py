import torch
from torch.utils.data import Dataset
from helper.utils import tokenize_smiles

class ReactionDataset(Dataset):
    def __init__(self, df, token2idx, max_len=100):
        self.df = df
        self.token2idx = token2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def encode(self, smiles, add_special=True):
        tokens = tokenize_smiles(smiles)
        if add_special:
            tokens = ['<sos>'] + tokens + ['<eos>']
        ids = [self.token2idx.get(tok, self.token2idx['<unk>']) for tok in tokens]
        padded = ids[:self.max_len] + [self.token2idx['<pad>']] * (self.max_len - len(ids))
        return torch.tensor(padded[:self.max_len])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.encode(row['reactants'], add_special=False)
        tgt = self.encode(row['products'], add_special=True)
        return src, tgt
