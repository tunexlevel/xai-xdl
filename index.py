from flask import Flask, jsonify, request
from predict.api_prediction import predict_product, _remove_atom_mapping
from predict.api_prediction_retro import (
    _remove_atom_mapping_labels,
    predict_reactants,
)
from explain import explain_prediction
from protocol.index import (
    protocol_to_xdl,
    reaction_to_draft_protocol,
    recipe_to_protocol,
)


app = Flask(__name__)

MAX_PREDICTIONS = 5


def _call_predictor(predictor, value, top_k, model_name=None):
    kwargs = {"top_k": top_k}

    if model_name:
        kwargs["file_name"] = model_name

    try:
        return predictor(value, **kwargs)
    except TypeError:
        kwargs.pop("file_name", None)
        return predictor(value, **kwargs)


def _normalise_predictions(raw_predictions, top_k):
    if raw_predictions is None:
        return []

    if not isinstance(raw_predictions, (list, tuple)):
        raw_predictions = [raw_predictions]

    results = []

    for item in raw_predictions[:MAX_PREDICTIONS]:
        prediction = item
        weight = None
        confidence = None
        source_tokens = []
        target_tokens = []

        if isinstance(item, dict):
            prediction = (
                item.get("prediction")
                or item.get("product_smiles")
                or item.get("reactant_smiles")
                or item.get("reactants")
                or item.get("smiles")
            )
            weight = item.get("weight", item.get("score", item.get("probability")))
            confidence = item.get("confidence")
            attention_weights = item.get(
                "attention_weights",
                item.get("attention"),
            )
            source_tokens = item.get("source_tokens", [])
            target_tokens = item.get("target_tokens", [])

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            prediction, weight = item[0], item[1]
            attention_weights = item[2] if len(item) >= 3 else None

        if prediction is None:
            continue

        prediction = _remove_atom_mapping_labels(str(prediction))
        if isinstance(source_tokens, list):
            source_tokens = [
                _remove_atom_mapping_labels(token) for token in source_tokens
            ]
        if isinstance(target_tokens, list):
            target_tokens = [
                _remove_atom_mapping_labels(token) for token in target_tokens
            ]

        try:
            weight = float(weight) if weight is not None else None
        except (TypeError, ValueError):
            weight = None

        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        results.append({
            "prediction": prediction,
            "weight": weight,
            "confidence": confidence,
            "attention_weights": attention_weights,
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
        })

    if not results:
        return []

    # Use uniform weights when the model does not return scores.
    supplied_weights = [r["weight"] for r in results]
    scalar_weights = [
        value for value in supplied_weights
        if isinstance(value, (int, float))
    ]

    if scalar_weights:
        total = sum(max(0.0, float(value)) for value in scalar_weights)
        normalised_weights = [
            round(max(0.0, float(value)) / total, 6)
            if isinstance(value, (int, float)) and total > 0
            else None
            for value in supplied_weights
        ]
    else:
        normalised_weights = [None] * len(results)

    for result, normalised_weight in zip(results, normalised_weights):
        if normalised_weight is not None:
            result["weight"] = normalised_weight

        if result["confidence"] is None:
            result["confidence"] = normalised_weight

        if result["confidence"] is not None:
            result["confidence"] = round(
                max(0.0, min(1.0, float(result["confidence"]))),
                6,
            )

    return results[:top_k]


def _get_top_k():
    data = request.get_json(silent=True) or {}
    try:
        return max(1, min(int(data.get("top_k", 5)), MAX_PREDICTIONS))
    except (TypeError, ValueError):
        return MAX_PREDICTIONS


def _json_input(field_name):
    data = request.get_json(silent=True) or {}
    value = data.get(field_name)

    if not isinstance(value, str) or not value.strip():
        return None, jsonify({
            "status": 400,
            "error": f"'{field_name}' is required and must be a non-empty string"
        }), 400

    return value.strip(), None, None


def _get_model_name():
    data = request.get_json(silent=True) or {}
    return (
        data.get("model_name")
        or data.get("model")
        or data.get("file_name")
    )


def _protocol_request():
    data = request.get_json(silent=True) or {}
    include_protocol = data.get("include_protocol", False)
    if isinstance(include_protocol, str):
        include_protocol = include_protocol.strip().lower() in {"1", "true", "yes"}
    return bool(include_protocol), data.get("recipe_text")


def _explanation_requested():
    data = request.get_json(silent=True) or {}
    value = data.get("include_explanation", False)
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _attach_explanation(prediction, reactants, product):
    return explain_prediction(
        reactants,
        product,
        source_tokens=prediction.get("source_tokens"),
        target_tokens=prediction.get("target_tokens"),
        attention_weights=prediction.get("attention_weights"),
    )


def _attach_protocol(recipe_text, reactant_smiles, product_smiles, run_name="from_api"):
    if isinstance(recipe_text, str) and recipe_text.strip():
        steps = recipe_to_protocol(recipe_text)
        warnings = [
            "Protocol is generated from supplied recipe text and requires human approval."
        ]
    else:
        steps, warnings = reaction_to_draft_protocol(
            reactant_smiles,
            product_smiles=product_smiles,
        )
    xdl = protocol_to_xdl(steps, run_name=run_name)
    return {
        "status": "needs_review",
        "steps": steps,
        "xdl": xdl,
        "warnings": warnings,
    }


@app.post("/generate/xdl")
def generate_xdl():
    """Generate validated XDL from recipe text or reactant SMILES."""
    data = request.get_json(silent=True) or {}
    recipe_text = data.get("recipe_text")
    reactant_smiles = data.get("reactant_smiles", "")
    product_smiles = data.get("product_smiles", "")
    run_name = str(data.get("run_name") or "xdl_generation").strip()

    if not isinstance(recipe_text, str) or not recipe_text.strip():
        recipe_text = None
    if not isinstance(reactant_smiles, str):
        reactant_smiles = ""
    if not isinstance(product_smiles, str):
        product_smiles = ""

    if recipe_text is None and not reactant_smiles.strip():
        return jsonify({
            "status": "error",
            "error": "Provide either 'recipe_text' or 'reactant_smiles'.",
        }), 400

    try:
        protocol = _attach_protocol(
            recipe_text,
            reactant_smiles,
            product_smiles,
            run_name=run_name or "xdl_generation",
        )
    except (RuntimeError, ValueError) as error:
        return jsonify({
            "status": "error",
            "error": str(error),
        }), 400

    return jsonify(protocol)


@app.get("/")
def home():
    return jsonify({
        "status": 200,
        "message": "XAI XDL prediction API",
        "endpoints": {
            "forward": "POST /predict/forward",
            "retrosynthesis": "POST /predict/retrosynthesis",
            "xdl": "POST /generate/xdl",
            "health": "GET /health",
        },
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "The XAI XDL API is up and running!"
    })


@app.post("/predict/forward")
def forward_prediction():
    reactant_smiles, error, status = _json_input("reactant_smiles")
    if error:
        return error, status

    top_k = _get_top_k()
    raw_predictions = _call_predictor(
        predict_product,
        reactant_smiles,
        top_k,
        _get_model_name(),
    )

    include_protocol, recipe_text = _protocol_request()
    predictions = _normalise_predictions(raw_predictions, top_k)
    if _explanation_requested():
        for prediction in predictions:
            prediction["explanation"] = _attach_explanation(
                prediction,
                reactant_smiles,
                prediction["prediction"],
            )
    response = {
        "task": "forward_prediction",
        "reactant_smiles": _remove_atom_mapping(reactant_smiles),
        "predictions": predictions,
    }
    if include_protocol:
        try:
            prediction_smiles = predictions[0]["prediction"] if predictions else ""
            protocol = _attach_protocol(
                recipe_text,
                reactant_smiles,
                prediction_smiles,
                run_name="forward_prediction",
            )
        except (RuntimeError, ValueError) as error:
            return jsonify({
                **response,
                "protocol": {
                    "status": "needs_review",
                    "steps": [],
                    "xdl": None,
                    "warnings": [str(error)],
                },
            })
        for prediction in predictions:
            prediction["protocol"] = {
                **protocol,
                "product_smiles": prediction["prediction"],
            }
        response["protocol"] = protocol
    return jsonify(response)


@app.post("/predict/retrosynthesis")
def retrosynthesis_prediction():
    if predict_reactants is None:
        return jsonify({
            "status": 501,
            "error": (
                "Retrosynthesis is not implemented. "
                "Add predict_reactants to controller."
            ),
        }), 501

    product_smiles, error, status = _json_input("product_smiles")
    if error:
        return error, status

    top_k = _get_top_k()
    raw_predictions = _call_predictor(
        predict_reactants,
        product_smiles,
        top_k,
        _get_model_name(),
    )

    include_protocol, recipe_text = _protocol_request()
    predictions = _normalise_predictions(raw_predictions, top_k)
    if _explanation_requested():
        for prediction in predictions:
            prediction["explanation"] = _attach_explanation(
                prediction,
                prediction["prediction"],
                product_smiles,
            )
    response = {
        "task": "retrosynthesis",
        "product_smiles": _remove_atom_mapping(product_smiles),
        "predictions": predictions,
    }
    if include_protocol:
        try:
            predicted_reactants = predictions[0]["prediction"] if predictions else ""
            protocol = _attach_protocol(
                recipe_text,
                predicted_reactants,
                product_smiles,
                run_name="retrosynthesis",
            )
        except (RuntimeError, ValueError) as error:
            return jsonify({
                **response,
                "protocol": {
                    "status": "needs_review",
                    "steps": [],
                    "xdl": None,
                    "warnings": [str(error)],
                },
            })
        for prediction in predictions:
            prediction["protocol"] = {
                **protocol,
                "product_smiles": product_smiles,
            }
        response["protocol"] = protocol

    return jsonify(response)


# Backward-compatible forward prediction endpoint.
@app.get("/predict/<path:reactant_smiles>")
def legacy_forward_prediction(reactant_smiles):
    raw_predictions = _call_predictor(
        predict_product,
        reactant_smiles,
        _get_top_k(),
    )

    return jsonify({
        "task": "forward_prediction",
        "reactant_smiles": reactant_smiles,
        "predictions": _normalise_predictions(raw_predictions, _get_top_k()),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)