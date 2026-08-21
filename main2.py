from data_loader import load_uspto_file
from utils import build_vocab
from dataset import ReactionDataset
from torch.utils.data import DataLoader

# Load sample data
df = load_uspto_file("data/uspto50k/raw_train.csv", max_samples=1000)

# Build vocabulary from both reactants and products
all_smiles = df['reactants'].tolist() + df['products'].tolist()
token2idx, idx2token = build_vocab(all_smiles)

# Create dataset and dataloader
dataset = ReactionDataset(df, token2idx, max_len=120)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Inspect one batch
for src, tgt in loader:
    print("Reactants (token IDs):", src.shape)
    print("Products (token IDs):", tgt.shape)
    break
