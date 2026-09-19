from flask import Flask, jsonify, request
from predict.api_prediction import predict_product

try:
    from predict.predict import predict_reactants
except ImportError:
    predict_reactants = None

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

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            prediction, weight = item[0], item[1]
            attention_weights = item[2] if len(item) >= 3 else None

        if prediction is None:
            continue

        try:
            weight = float(weight) if weight is not None else None
        except (TypeError, ValueError):
            weight = None

        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        results.append({
            "prediction": str(prediction),
            "weight": weight,
            "confidence": confidence,
            "attention_weights": attention_weights,
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
    try:
        return max(1, min(int(request.args.get("top_k", 5)), MAX_PREDICTIONS))
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
        request.args.get("model")
        or request.args.get("file_name")
        or data.get("model")
        or data.get("file_name")
    )


@app.get("/")
def home():
    return jsonify({
        "status": 200,
        "message": "XAI XDL prediction API",
        "endpoints": {
            "forward": "POST /predict/forward",
            "retrosynthesis": "POST /predict/retrosynthesis",
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

    return jsonify({
        "task": "forward_prediction",
        "reactant_smiles": reactant_smiles,
        "predictions": _normalise_predictions(raw_predictions, top_k),
    })


@app.post("/predict/retrosynthesis")
def retrosynthesis_prediction():
    if predict_reactants is None:
        return jsonify({
            "status": 501,
            "error": (
                "Retrosynthesis is not implemented. "
                "Add predict_reactants to predict/predict.py."
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

    return jsonify({
        "task": "retrosynthesis",
        "product_smiles": product_smiles,
        "predictions": _normalise_predictions(raw_predictions, top_k),
    })


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
    app.run(host="0.0.0.0", port=5000, debug=True)