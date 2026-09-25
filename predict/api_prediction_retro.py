import json
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import torch
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.utils import (
    decode_indices,
    map_smiles,
    strip_atom_mapping,
    tokenize_smiles,
    valid_smiles_or_empty,
    get_single_root_aligned_pair,
    _canonicalize_reactants,
    canonicalize_preserve_molecule_order
)
from mod.model_retro import Seq2SeqTransformer


warnings.filterwarnings(
    "ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_FILE_NAME = "uspto50k_mapped_retro_3-3"


def _safe_file_name(file_name):
    file_name = str(file_name or DEFAULT_FILE_NAME).strip()
    if (
        not file_name
        or Path(file_name).name != file_name
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", file_name)
    ):
        raise ValueError("Invalid model file name")
    return file_name


def _is_mapped_model(file_name):
    return "unmapped" not in file_name.lower().replace("-", "_")


def _prepare_product(smiles, mapped):
    smiles = map_smiles(smiles)
    smiles = get_single_root_aligned_pair(smiles)
    return smiles


def _num_layers_from_file_name(file_name):
    match = re.search(r"(?:ed|retro)_(\d+)-\d+", file_name)
    return int(match.group(1)) if match else 3


@lru_cache(maxsize=8)
def _load_model(file_name):
    file_name = _safe_file_name(file_name)
    model_stem = f"{file_name}"
    model_path = ROOT / "pt" / "dump" / f"{file_name}_reaction_model.pt"
    token2idx_path = ROOT / "tokens" / "dump" / f"{model_stem}_token2idx.json"
    idx2token_path = ROOT / "tokens" / "dump" / f"{model_stem}_idx2token.json"

    with open(token2idx_path, "r") as file:
        model_token2idx = json.load(file)
    with open(idx2token_path, "r") as file:
        model_idx2token = {int(key): value for key, value in json.load(file).items()}

    pad_idx = model_token2idx.get("<pad>", 0)
    layers = _num_layers_from_file_name(file_name)
    model = Seq2SeqTransformer(
        input_dim=len(model_token2idx),
        output_dim=len(model_token2idx),
        emb_dim=256,
        nhead=8,
        num_encoder_layers=layers,
        num_decoder_layers=layers,
        dim_feedforward=512,
        pad_idx=pad_idx,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()

    return {
        "model": model,
        "token2idx": model_token2idx,
        "idx2token": model_idx2token,
        "pad_idx": pad_idx,
        "sos_idx": model_token2idx.get("<sos>", 1),
        "eos_idx": model_token2idx.get("<eos>", 2),
        "unk_idx": model_token2idx.get("<unk>"),
        "mapped": _is_mapped_model(file_name),
    }


def _normalise_scores(scores):
    if not scores:
        return []
    values = torch.tensor([float(score) for score in scores], dtype=torch.float32)
    return [round(float(value), 6) for value in torch.softmax(values, dim=0).tolist()]


def _round_nested(value, decimals=4):
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [_round_nested(item, decimals) for item in value]
    if isinstance(value, tuple):
        return [_round_nested(item, decimals) for item in value]
    return value


def _trim_special_attention_rows(attention_weights, target_length):
    if not isinstance(attention_weights, list):
        return attention_weights
    return attention_weights[:target_length]


def _remove_atom_mapping_labels(smiles):
    return re.sub(r":\d+(?=\])", "", smiles)


def _prepare_display_smiles(decoded_smiles):
    unmapped_smiles = _remove_atom_mapping_labels(decoded_smiles)
    unmapped_tokens = tokenize_smiles(unmapped_smiles)
    display_smiles = strip_atom_mapping(decoded_smiles, canonical=False)
    display_tokens = tokenize_smiles(display_smiles)

    if len(display_tokens) != len(unmapped_tokens):
        return unmapped_smiles, unmapped_tokens
    return display_smiles, display_tokens


def predict_reactants(
    product_smiles,
    top_k=5,
    max_len=120,
    file_name=DEFAULT_FILE_NAME,
):
    if not isinstance(product_smiles, str) or not product_smiles.strip():
        return []

    try:
        top_k = max(1, min(int(top_k), 5))
        bundle = _load_model(file_name)
        
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise RuntimeError(f"Unable to load model '{file_name}': {error}") from error

    product_smiles = _prepare_product(product_smiles.strip(), bundle["mapped"])
    tokens = tokenize_smiles(product_smiles)
    tokens2 = tokenize_smiles(strip_atom_mapping(product_smiles, canonical=False))
    if not tokens:
        return []

    src_ids = [
        bundle["token2idx"].get(
            token,
            bundle["unk_idx"] if bundle["unk_idx"] is not None else bundle["pad_idx"],
        )
        for token in tokens
    ]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        beam_candidates = bundle["model"].beam_search_candidates(
            src_tensor,
            bundle["sos_idx"],
            bundle["eos_idx"],
            beam_width=top_k,
            max_len=max_len,
            return_attention=True,
        )

    candidates = []
    seen = set()
    for sequence, score, attention_weights in beam_candidates:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().tolist()
        decoded_tokens = decode_indices(
            sequence,
            bundle["idx2token"],
            bundle["sos_idx"],
            bundle["eos_idx"],
            bundle["pad_idx"],
        )
        decoded_smiles = "".join(decoded_tokens)
        canonical_smiles = valid_smiles_or_empty(decoded_smiles)
        if not canonical_smiles:
            continue

        smiles, target_tokens = _prepare_display_smiles(decoded_smiles)
        attention_weights = _round_nested(
            _trim_special_attention_rows(attention_weights, len(target_tokens)),
        )
        canonical_prediction = strip_atom_mapping(canonical_smiles, canonical=True)
        if not smiles or canonical_prediction in seen:
            continue
        seen.add(canonical_prediction)
        candidates.append({
            "prediction": smiles,
            "canonical_prediction": canonical_prediction,
            "score": float(score),
            "attention_weights": attention_weights,
            "source_tokens": tokens2,
            "target_tokens": target_tokens,
            "model": file_name,
            "mapped_input": bundle["mapped"],
        })
        if len(candidates) >= top_k:
            break

    confidences = _normalise_scores([candidate["score"] for candidate in candidates])
    for candidate, confidence in zip(candidates, confidences):
        candidate["confidence"] = confidence
        candidate["weight"] = confidence
    return candidates


def main(
    product_smiles,
    top_k=5,
    max_len=120,
    target_smiles=None,
):
    predictions = predict_reactants(
        product_smiles=product_smiles,
        top_k=top_k,
        max_len=max_len,
        #uspto50k_unmapped_ed_6-6
        #uspto50k_unmapped_lr_5e-4
        #uspto50k_mapped_lr_5e-4
        #uspto50k_mapped_lr_3e-4
        file_name='uspto50k_mapped_retro_3-4'
    )

    print(f"Predictions for {strip_atom_mapping(product_smiles)}:")
    for i, result in enumerate(predictions):
        print(f"{i + 1}. {result['prediction']} (confidence: {result['confidence']})")  
        


if __name__ == "__main__":
    input_smiles = input("Enter product SMILES: ").strip()
    if not input_smiles:
        print("No input provided. Using default product SMILES.")
        input_smiles = "[C:1](=[O:2])([CH2:3][c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1)[NH:14][c:13]1[n:12][c:11]([CH3:10])[c:16]([C:17](=[O:18])[NH:19][CH2:20][c:21]2[cH:22][cH:23][cH:24][cH:25][cH:26]2)[s:15]1"
        # print(input_smiles)
    main(product_smiles=input_smiles, top_k=10)