import json
import sys
import warnings

from pathlib import Path

import torch
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mod.model import Seq2SeqTransformer
from helper.utils import (
    decode_indices,
    strip_atom_mapping_labels,
    tokenize_smiles,
    valid_smiles_or_empty,
)
import re
from functools import lru_cache

from rdkit import Chem


warnings.filterwarnings(
    "ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_FILE_NAME = "uspto50k_unmapped_lr_5e-4"


def _safe_file_name(file_name):
    """Prevent paths from being supplied as model names."""
    file_name = str(file_name or DEFAULT_FILE_NAME).strip()

    if (
        not file_name
        or Path(file_name).name != file_name
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", file_name)
    ):
        raise ValueError("Invalid model file name")

    return file_name


def _is_mapped_model(file_name):
    name = file_name.lower()
    return "unmapped" not in name and "unmapped" not in name.replace("-", "_")


def _has_atom_mapping(smiles):
    return bool(re.search(r":\d+\]", smiles))


def _prepare_reactants(smiles, mapped):
    """Add or remove atom-map numbers according to the selected model."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles

    if mapped:
        if not _has_atom_mapping(smiles):
            for atom_number, atom in enumerate(molecule.GetAtoms(), start=1):
                atom.SetAtomMapNum(atom_number)
    else:
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        molecule,
        canonical=False,
        isomericSmiles=True,
    )


def _num_layers_from_file_name(file_name):
    """Extract the number of encoder/decoder layers from the file name."""
    match = re.search(r"ed_(\d+)-(\d+)", file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 3, 3

@lru_cache(maxsize=8)
def _load_model(file_name):
    file_name = _safe_file_name(file_name)

    model_path = ROOT / "pt" / "dump" / f"{file_name}_reaction_model.pt"
    token2idx_path = ROOT / "tokens" / "dump" / f"{file_name}_token2idx.json"
    idx2token_path = ROOT / "tokens" / "dump" / f"{file_name}_idx2token.json"

    with open(token2idx_path, "r") as file:
        model_token2idx = json.load(file)

    with open(idx2token_path, "r") as file:
        model_idx2token = {
            int(key): value
            for key, value in json.load(file).items()
        }

    model_pad_idx = model_token2idx.get("<pad>", 0)

    model = Seq2SeqTransformer(
        input_dim=len(model_token2idx),
        output_dim=len(model_token2idx),
        emb_dim=256,
        nhead=8,
        num_encoder_layers=_num_layers_from_file_name(file_name)[0],
        num_decoder_layers=_num_layers_from_file_name(file_name)[1],
        dim_feedforward=512,
        pad_idx=model_pad_idx,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    return {
        "model": model,
        "token2idx": model_token2idx,
        "idx2token": model_idx2token,
        "pad_idx": model_pad_idx,
        "sos_idx": model_token2idx.get("<sos>", 1),
        "eos_idx": model_token2idx.get("<eos>", 2),
        "unk_idx": model_token2idx.get("<unk>"),
        "mapped": _is_mapped_model(file_name),
    }


def _round_nested(value, decimals=4):
    """Round all numeric values in nested attention data."""
    if isinstance(value, float):
        return round(value, decimals)

    if isinstance(value, int):
        return value

    if isinstance(value, list):
        return [_round_nested(item, decimals) for item in value]

    if isinstance(value, tuple):
        return [_round_nested(item, decimals) for item in value]

    return value


def _to_json(value):
    """Convert tensors and arrays to JSON-compatible values."""
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)

    return _round_nested(value, decimals=4)


def _unpack_candidate(candidate):
    """
    Supports model outputs in these formats:

        (sequence, score, attention_weights)
        {"sequence": ..., "score": ..., "attention_weights": ...}
    """
    if isinstance(candidate, dict):
        sequence = (
            candidate.get("sequence")
            or candidate.get("tokens")
            or candidate.get("prediction")
        )
        score = candidate.get(
            "score",
            candidate.get("log_probability", candidate.get("probability", 0.0)),
        )
        attention = candidate.get(
            "attention_weights",
            candidate.get("attention", candidate.get("attentions")),
        )
        return sequence, score, _to_json(attention)

    if isinstance(candidate, (list, tuple)):
        sequence = candidate[0] if len(candidate) > 0 else None
        score = candidate[1] if len(candidate) > 1 else 0.0
        attention = candidate[2] if len(candidate) > 2 else None
        return sequence, score, _to_json(attention)

    return candidate, 0.0, None


def _run_beam_search(model, src_tensor, sos_idx, eos_idx, top_k, max_len):
    try:
        return model.beam_search_candidates(
            src_tensor,
            sos_idx,
            eos_idx,
            beam_width=top_k,
            max_len=max_len,
            return_attention=True,
        )
    except TypeError:
        return model.beam_search_candidates(
            src_tensor,
            sos_idx,
            eos_idx,
            beam_width=top_k,
            max_len=max_len,
        )


def _remove_atom_mapping(smiles):
    """Remove atom-map numbers from predicted SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        molecule,
        canonical=False,
        isomericSmiles=True,
    )


def _normalise_scores(scores):
    if not scores:
        return []

    values = torch.tensor(
        [float(score) for score in scores],
        dtype=torch.float32,
    )

    return [
        round(float(value), 6)
        for value in torch.softmax(values, dim=0).tolist()
    ]


def _trim_special_attention_rows(attention_weights, target_length):
    """Keep decoder rows that predict target tokens, starting with the SOS row."""
    if not isinstance(attention_weights, list):
        return attention_weights
    return attention_weights[:target_length]


def predict_product(
    reactant_smiles,
    top_k=5,
    max_len=120,
    target_smiles=None,
    file_name=DEFAULT_FILE_NAME,
):
    if not isinstance(reactant_smiles, str) or not reactant_smiles.strip():
        return []

    try:
        bundle = _load_model(file_name)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise RuntimeError(
            f"Unable to load model '{file_name}': {error}"
        ) from error

    try:
        top_k = max(1, min(int(top_k), 5))
    except (TypeError, ValueError):
        top_k = 5

    reactant_smiles = _prepare_reactants(
        reactant_smiles.strip(),
        mapped=bundle["mapped"],
    )
    
    model_tokens = tokenize_smiles(reactant_smiles)
    source_tokens = [
        strip_atom_mapping_labels(token)
        for token in model_tokens
    ]
    if not model_tokens:
        return []

    src_ids = [
        bundle["token2idx"].get(
            token,
            bundle["unk_idx"]
            if bundle["unk_idx"] is not None
            else bundle["pad_idx"],
        )
        for token in model_tokens
    ]

    src_tensor = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        beam_candidates = _run_beam_search(
            bundle["model"],
            src_tensor,
            bundle["sos_idx"],
            bundle["eos_idx"],
            top_k,
            max_len,
        )

    candidates = []
    seen = set()
    
    for candidate in beam_candidates:
        sequence, score, attention_weights = _unpack_candidate(candidate)

        if sequence is None:
            continue

        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().tolist()

        decoded_tokens = decode_indices(
            sequence,
            bundle["idx2token"],
            bundle["sos_idx"],
            bundle["eos_idx"],
            bundle["pad_idx"],
        )
        smiles = valid_smiles_or_empty("".join(decoded_tokens))
        if not smiles:
            continue

        target_tokens = [
            strip_atom_mapping_labels(token) for token in decoded_tokens
        ]
        attention_weights = _trim_special_attention_rows(
            attention_weights,
            len(target_tokens),
        )
        
        # Always expose unmapped predictions through the API.
        smiles = _remove_atom_mapping(smiles)
        if not smiles or smiles in seen:
            continue

        seen.add(smiles)

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        candidates.append({
            "prediction": smiles,
            "score": score,
            "attention_weights": attention_weights,
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
        })

        if len(candidates) >= top_k:
            break

    if not candidates:
        return []

    confidences = _normalise_scores(
        [candidate["score"] for candidate in candidates]
    )

    return [
        {
            "prediction": candidate["prediction"],
            "weight": candidate["attention_weights"],
            "attention_weights": candidate["attention_weights"],
            "source_tokens": candidate["source_tokens"],
            "target_tokens": candidate["target_tokens"],
            "confidence": confidence,
            "model": file_name,
            "mapped_input": bundle["mapped"],
        }
        for candidate, confidence in zip(candidates, confidences)
    ]
    
    


def main(
    reactant_smiles="O=C(O[C:1](=[O:2])[C:3]([F:4])([F:5])[F:6])C(F)(F)F.[NH2:7][CH2:8][c:9]1[cH:10][cH:11][cH:12][cH:13][c:14]1[S:15](=[O:16])(=[O:17])[CH:18]1[CH2:19][CH2:20]1",
    top_k=5,
    max_len=120,
    target_smiles=None,
):
    predictions = predict_product(
        reactant_smiles=reactant_smiles,
        top_k=top_k,
        max_len=max_len,
        target_smiles=target_smiles,
        #uspto50k_unmapped_ed_6-6
        #uspto50k_unmapped_lr_5e-4
        #uspto50k_mapped_lr_5e-4
        #uspto50k_mapped_lr_3e-4
        file_name='uspto50k_mapped_lr_5e-4', 
    )

    print(f"Predictions for {reactant_smiles}:")
    for i, result in enumerate(predictions):
        print(f"{i + 1}. {result['prediction']} (confidence: {result['confidence']})")  
        


if __name__ == "__main__":
    input_smiles = input("Enter reactant SMILES: ").strip()
    if not input_smiles:
        print("No input provided. Using default reactant SMILES.")
        input_smiles = "O=C(O[C:1](=[O:2])[C:3]([F:4])([F:5])[F:6])C(F)(F)F.[NH2:7][CH2:8][c:9]1[cH:10][cH:11][cH:12][cH:13][c:14]1[S:15](=[O:16])(=[O:17])[CH:18]1[CH2:19][CH2:20]1"
    main(reactant_smiles=input_smiles, top_k=1)