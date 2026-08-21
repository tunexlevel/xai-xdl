

from rdkit import Chem
import torch

def atom_features(atom):
    # simple features: one-hot atomic number (or extend)
    return torch.tensor([
        atom.GetAtomicNum(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge()
    ], dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = []
    edge_index = [[], []]
    for i, a in enumerate(mol.GetAtoms()):
        node_feats.append(atom_features(a))
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
    x = torch.stack(node_feats, dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x, edge_index

