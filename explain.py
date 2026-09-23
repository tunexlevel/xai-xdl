import re
from typing import Any


REACTION_CUES = (
    ("aryl formylation", ("CN(C)C=O",), ("Br", "Cl", "I")),
    ("aryl amination / substitution", ("N",), ("Br", "Cl", "I")),
    ("Suzuki-type cross-coupling", ("B(O)O", "B(O", "B1"), ("Br", "Cl", "I")),
    ("reduction of a nitro group", ("[N+](=O)[O-]", "N(=O)=O"), ()),
    ("acylation / amidation", ("C(=O)",), ("N",)),
    ("oxidation or reduction involving a carbonyl", ("C(=O)",), ("O",)),
)


def explain_prediction(
    reactant_smiles: str,
    product_smiles: str,
    source_tokens: list[str] | None = None,
    target_tokens: list[str] | None = None,
    attention_weights: Any = None,
) -> dict[str, Any]:
    """Build an evidence-first, LLM-ready explanation for one prediction."""
    source_tokens = list(source_tokens or [])
    target_tokens = list(target_tokens or [])
    reaction_class, class_confidence, class_evidence = _infer_reaction_class(
        reactant_smiles,
        product_smiles,
    )
    attention = _summarize_attention(
        source_tokens,
        target_tokens,
        attention_weights,
    )

    explanation = _build_explanation_text(
        reaction_class,
        class_confidence,
        class_evidence,
        attention,
    )
    return {
        "method": "heuristic_features_and_attention",
        "reaction_class": reaction_class,
        "reaction_class_confidence": class_confidence,
        "reaction_class_evidence": class_evidence,
        "attention_summary": attention,
        "explanation": explanation,
        "limitations": [
            "Attention weights show token associations, not proven chemical causality.",
            "Reaction class is a heuristic until a trained reaction classifier or reviewed procedure data is added.",
            "Mechanistic interpretation requires conditions, reagents, and experimental context.",
        ],
        "llm_input": {
            "reactants": reactant_smiles,
            "product": product_smiles,
            "reaction_class": reaction_class,
            "class_evidence": class_evidence,
            "attention_summary": attention,
        },
    }


def _infer_reaction_class(reactants: str, product: str) -> tuple[str, str, list[str]]:
    text = f"{reactants}>>{product}"
    for name, required, supporting in REACTION_CUES:
        if all(marker in reactants for marker in required):
            evidence = [f"reactants contain {marker}" for marker in required]
            evidence.extend(
                f"reactants contain {marker}" for marker in supporting if marker in reactants
            )
            if name == "aryl formylation" and (
                "C(=O)" in product or "O=C" in product
            ):
                evidence.append("predicted product contains a carbonyl")
                return name, "low", evidence
            if name != "aryl formylation":
                return name, "low", evidence

    if "." in reactants:
        return (
            "multi-component transformation",
            "low",
            ["reactants contain multiple dot-separated components"],
        )
    return "unclassified transformation", "low", []


def _summarize_attention(
    source_tokens: list[str],
    target_tokens: list[str],
    attention_weights: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    matrix = _numeric_matrix(attention_weights)
    if not matrix or not source_tokens or not target_tokens:
        return {
            "available": False,
            "orientation": "target_token_to_source_token",
            "top_source_tokens": [],
            "target_token_links": [],
        }

    rows = min(len(matrix), len(target_tokens))
    cols = min(len(matrix[0]), len(source_tokens)) if matrix[0] else 0
    if not rows or not cols:
        return {
            "available": False,
            "orientation": "target_token_to_source_token",
            "top_source_tokens": [],
            "target_token_links": [],
        }

    totals = [0.0] * cols
    links = []
    for target_index in range(rows):
        row = matrix[target_index][:cols]
        total = sum(max(0.0, value) for value in row)
        if total <= 0:
            continue
        ranked = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:top_k]
        for source_index, value in ranked:
            totals[source_index] += max(0.0, value)
        links.append({
            "target_token": target_tokens[target_index],
            "target_index": target_index,
            "source_tokens": [
                {
                    "token": source_tokens[source_index],
                    "source_index": source_index,
                    "weight": round(float(value / total), 6),
                }
                for source_index, value in ranked
                if value > 0
            ],
        })

    ranked_totals = sorted(enumerate(totals), key=lambda item: item[1], reverse=True)
    denominator = sum(value for _, value in ranked_totals) or 1.0
    return {
        "available": bool(links),
        "orientation": "target_token_to_source_token",
        "top_source_tokens": [
            {
                "token": source_tokens[index],
                "source_index": index,
                "aggregate_weight": round(float(value / denominator), 6),
            }
            for index, value in ranked_totals[:top_k]
            if value > 0
        ],
        "target_token_links": links,
    }


def _numeric_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        value = value[0]
    matrix = []
    for row in value:
        if not isinstance(row, list):
            continue
        numbers = []
        for item in row:
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                numbers.append(0.0)
        if numbers:
            matrix.append(numbers)
    return matrix


def _build_explanation_text(
    reaction_class: str,
    confidence: str,
    evidence: list[str],
    attention: dict[str, Any],
) -> str:
    parts = [
        f"The predicted transformation is provisionally classified as {reaction_class} ({confidence} confidence)."
    ]
    if evidence:
        parts.append("Class evidence: " + "; ".join(evidence) + ".")
    if attention.get("available"):
        tokens = [item["token"] for item in attention["top_source_tokens"]]
        parts.append(
            "The prediction attended most strongly to source tokens: "
            + ", ".join(repr(token) for token in tokens)
            + "."
        )
    else:
        parts.append("No usable attention matrix was available for token-level evidence.")
    return " ".join(parts)
