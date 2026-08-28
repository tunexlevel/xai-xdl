from collections import Counter
from rdkit import Chem
import re

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"



TOKEN_REGEX = r"""
    Cl|Br|Si|Se|Na|Li|Mg|Ca|Al|Fe|Cu|Zn|Sn|Pb|Ag|Au|Co|Mn|Ti|Cr|Pt|Hg|Ni|
    \%\d{2} |        # ring numbers like %12
    \[[^\]]+\] |     # bracket expressions: [C@H], [n+], [O-], [13C], etc.
    \=|\#|\-|\+|\\|\/|\. |
    \(|\)|\[|\] |
    [A-Za-z] |       # single atom symbols (C, N, O, B, P, S, F, I, ...)
    \d               # ring digits
"""

#token_regex = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)| \.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

token_pattern = re.compile(TOKEN_REGEX, re.X)


def tokenize_smiles(smiles: str):
    if not smiles or not isinstance(smiles, str):
        return []

    smiles = smiles.strip()

    tokens = token_pattern.findall(smiles)
    return tokens



def strip_atom_mapping(smi: str):
    """
    Removes atom mapping numbers from SMILES using RDKit.
    Returns the original string if RDKit cannot parse it.
    """
    if not smi or not isinstance(smi, str):
        return ""

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi  # keep original to avoid destroying unparseable SMILES

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol, canonical=True)


def clean_and_tokenize(smiles):
    smiles = strip_atom_mapping(smiles)
    return tokenize_smiles(smiles)


def build_vocab(smiles_list, file_type=None):
    tokens = []

    for smi in smiles_list:
        #clean_smi = strip_atom_mapping(smi)
        tokens.extend(tokenize_smiles(smi))

    counter = Counter(tokens)
    sorted_tokens = sorted(counter.keys())

    token2idx = {
        PAD_TOKEN: 0,
        SOS_TOKEN: 1,
        EOS_TOKEN: 2,
        UNK_TOKEN: 3,
    }
    idx2token = {v: k for k, v in token2idx.items()}

    for tok in sorted_tokens:
        if tok not in token2idx:
            idx = len(token2idx)
            token2idx[tok] = idx
            idx2token[idx] = tok

    return token2idx, idx2token


def decode_indices(indices, idx2token, sos_idx=1, eos_idx=2, pad_idx=0):
    result = []
    for idx in indices:
        if idx == sos_idx:
            continue
        if idx == eos_idx:
            break
        if idx == pad_idx:
            continue
        token = idx2token.get(int(idx))
        if token is not None:
            result.append(token)
    return result


def valid_smiles_or_empty(smiles):
    if not isinstance(smiles, str):
        return ""
    cleaned = smiles.strip().replace("<pad>", "")
    if not cleaned:
        return ""
    try:
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""

