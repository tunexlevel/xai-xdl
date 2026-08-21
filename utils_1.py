from collections import Counter
from rdkit import Chem

def tokenize_smiles(smiles):
    """
    Tokenizes a SMILES string into a list of tokens.
    Handles 'Cl' and 'Br' as special cases.
    """
    tokens = []
    i = 0
    while i < len(smiles):
        if i+1 < len(smiles) and smiles[i:i+2] in ('Cl', 'Br'):
            tokens.append(smiles[i:i+2])
            i += 2
        else:
            tokens.append(smiles[i])
            i += 1
    return tokens

def build_vocab1(smiles_list, special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]):
    """
    Builds vocab from a list of SMILES strings.
    Returns token-to-index and index-to-token dictionaries.
    """
    counter = Counter()
    for smi in smiles_list:
        tokens = tokenize_smiles(smi)
        counter.update(tokens)

    all_tokens = special_tokens + sorted(counter.keys())
    token2idx = {token: int(idx) for idx, token in enumerate(all_tokens)}
    idx2token = {int(idx): token for token, idx in token2idx.items()}
    
    return token2idx, idx2token

def build_vocab(smiles_list):
    from collections import Counter

    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

    counter = Counter()
    for smi in smiles_list:
        counter.update(tokenize_smiles(smi))

    tokens = special_tokens + sorted(counter.keys())
    token2idx = {tok: idx for idx, tok in enumerate(tokens)}
    idx2token = {idx: tok for tok, idx in token2idx.items()}

    return token2idx, idx2token


# Tokenize and convert to indices, then pad
def smiles_to_indices(smiles, token2idx, max_len):
    tokens = ["<sos>"] + tokenize_smiles(smiles) + ["<eos>"]
    idxs = [token2idx.get(tok, token2idx["<unk>"]) for tok in tokens]
    idxs = idxs[:max_len] + [token2idx["<pad>"]] * (max_len - len(idxs))
    return idxs

def is_chemically_valid(smiles_string):
    """
    Checks if a SMILES string can be parsed into a chemically valid RDKit molecule object.
    
    Args:
        smiles_string (str): The predicted SMILES string (e.g., 'CCOc1cncc(Br)c1').
        
    Returns:
        bool: True if the molecule is valid, False otherwise.
    """
    try:
        # Attempt to create an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles_string, sanitize=True)
        # Check if mol is not None (parsing succeeded) AND has atoms (not an empty string)
        if mol is not None and mol.GetNumAtoms() > 0:
            return True
        else:
            return False
    except:
        # Catch any specific RDKit exceptions during parsing/sanitization
        return False

def calculate_structural_validity(predicted_candidates_list):
    """
    Calculates the validity for a list of predicted SMILES strings (e.g., Top-1 or Top-10).
    
    Args:
        predicted_candidates_list (list): A list of dictionaries or strings
                                        (e.g., the 'smiles' key from your beam search output).
                                        
    Returns:
        float: The percentage of valid predictions in the list.
    """
    valid_count = 0
    total_count = len(predicted_candidates_list)
    
    # Ensure you are working with just the SMILES strings, not the full dicts
    smiles_list = [d['smiles'] if isinstance(d, dict) else d for d in predicted_candidates_list]
    
    for smiles in smiles_list:
        if is_chemically_valid(smiles):
            valid_count += 1
            
    # Return the percentage
    return (valid_count / total_count) * 100 if total_count > 0 else 0.0